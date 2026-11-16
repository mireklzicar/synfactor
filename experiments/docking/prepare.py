"""Prepare label-free candidate/feature arrays and a separate frozen oracle.

The full-product fingerprint cache is RDKit path/2048, NOT Morgan. Its row
alignment is recovered from reaction/slot Cartesian ordering, then checked
against independently computed fingerprints of sampled product graphs.
"""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs

from chemistry import cap_synthon, assemble
from teachers import RDKitTeacher


def read_smiles_rows(table, indices):
    """Read only relevant Parquet row groups, bounded to one group at a time."""
    parquet = pq.ParquetFile(table)
    wanted = np.asarray(indices, np.int64)
    result, offset = {}, 0
    for group in range(parquet.metadata.num_row_groups):
        end = offset + parquet.metadata.row_group(group).num_rows
        selected = wanted[(wanted >= offset) & (wanted < end)]
        if len(selected):
            values = parquet.read_row_group(group, columns=["smiles"]).take(selected-offset)["smiles"].to_pylist()
            result.update(zip(selected.tolist(), values))
        offset = end
    return [result[int(i)] for i in wanted]


def audit_product_identity(cache, chunk_size=1000000):
    """Label-free ID audit: retain the first source row, as released .sidx does.

    Sorted 128-bit index keys reveal aliases without reading score values or
    sorting the full candidate table. Only the sparse alias map is retained.
    """
    cache = Path(cache)
    destination = cache / "product_aliases.json"
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.perf_counter()
    manifest = json.loads((cache / "manifest.json").read_text())
    pids = np.load(cache / "product_ids.npy", mmap_mode="r")
    index = Path(manifest["score_table"] + ".sidx")
    duplicate_ids, expected = set(), 0
    if (index / "keys_hi.npy").exists():
        hi = np.load(index / "keys_hi.npy", mmap_mode="r")
        lo = np.load(index / "keys_lo.npy", mmap_mode="r")
        if len(hi) != len(pids) or len(lo) != len(pids):
            raise ValueError("Identity index and candidate table lengths differ")
        previous = None
        for start in range(0, len(hi), chunk_size):
            h, l = hi[start:start+chunk_size], lo[start:start+chunk_size]
            repeated = np.flatnonzero((h[1:] == h[:-1]) & (l[1:] == l[:-1])) + 1
            pairs = [(int(h[j]), int(l[j])) for j in repeated]
            if previous == (int(h[0]), int(l[0])):
                pairs.append(previous)
            expected += len(pairs)
            duplicate_ids.update(f"{a:016x}{b:016x}".encode() for a, b in pairs)
            previous = (int(h[-1]), int(l[-1]))
        del hi, lo
    else:
        if len(pids) > 2000000:
            raise ValueError("Large identity audit requires released sorted .sidx keys")
        unique, counts = np.unique(pids, return_counts=True)
        duplicate_ids = set(unique[counts > 1].tolist())
        expected = int((counts-1).sum())
    duplicate_ids = np.array(sorted(duplicate_ids), dtype="S32")
    first, aliases = {}, {}
    if len(duplicate_ids):
        for start in range(0, len(pids), chunk_size):
            values = pids[start:start+chunk_size]
            positions = np.searchsorted(duplicate_ids, values)
            selected = np.flatnonzero((positions < len(duplicate_ids)) &
                       (duplicate_ids[np.minimum(positions, len(duplicate_ids)-1)] == values))
            for local in selected:
                row, pid = start + int(local), bytes(values[local])
                if pid in first:
                    aliases[str(row)] = first[pid]
                else:
                    first[pid] = row
    if len(aliases) != expected:
        raise ValueError("Cached candidate IDs disagree with released identity index")
    result = dict(version=1, policy="first source row per product_id, matching released .sidx",
                  source_rows=len(pids), unique_product_ids=len(pids)-len(aliases),
                  duplicate_product_ids=len(duplicate_ids), aliased_rows=len(aliases),
                  aliases=aliases, label_access=False,
                  identity_repair_seconds=time.perf_counter()-started)
    destination.write_text(json.dumps(result, indent=2)+"\n")
    manifest["product_identity"] = {k: v for k, v in result.items() if k != "aliases"}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Identity audit: {len(aliases):,} alias rows excluded; {result['identity_repair_seconds']:.1f}s", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("/home/mirekl/SynthonBench_ICML/hf_synthonbench_dataset"))
    p.add_argument("--fingerprints", type=Path, default=Path("/home/mirekl/SynthonBench_ICML/synthon_bench/data"))
    p.add_argument("--scale", default="1M", choices=["1M", "10M", "100M"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-fingerprints", action="store_true", help="For plain CP runs that need no product fingerprints")
    args = p.parse_args()
    if (args.out / "manifest.json").exists():
        audit_product_identity(args.out)
        print("Prepared cache already exists", flush=True); return
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    kind = "glide" if args.scale == "1M" else "surrogate"
    table = args.data / "scores" / f"{kind}_{args.scale}.parquet"
    space_path = args.data / "spaces" / f"synthon_space_{args.scale}.synthons.tsv"
    frame = pd.read_csv(space_path, sep="\t", dtype={"reaction_id": str})
    columns = ["product_id", "reaction_id", "synthon1_id", "synthon2_id", "synthon3_id"]
    # Structural filter is allowed only in separately labeled filter variants.
    if kind == "glide":
        columns += ["passes_exp27_exact_filter"]
    products = pq.read_table(table, columns=columns).to_pandas()
    for j in range(1, 4):
        products[f"synthon{j}_id"] = pd.to_numeric(products[f"synthon{j}_id"], errors="coerce").astype("Int64")
    n = len(products)
    slots = {str(r): [g.index[g.position == j].to_numpy() for j in sorted(g.position.unique())]
             for r, g in frame.groupby("reaction_id", sort=False)}
    keys = ["reaction_id", "position", "synthon_id"]
    unique_frame = frame.drop_duplicates(keys, keep="first")
    lookup = pd.MultiIndex.from_frame(unique_frame[keys])
    tuples = np.full((n, 3), -1, np.int32)
    for j in range(3):
        valid = products[f"synthon{j+1}_id"].notna().values
        lookup_keys = pd.MultiIndex.from_arrays([products.loc[valid, "reaction_id"], np.full(valid.sum(), j+1),
                                         products.loc[valid, f"synthon{j+1}_id"].astype(np.int64)])
        mapped = lookup.get_indexer(lookup_keys)
        if np.any(mapped < 0):
            raise ValueError("Unknown tuple synthon")
        tuples[valid, j] = unique_frame.index.values[mapped]
    # Some historical synthon IDs refer to two different attachment graphs.
    # Resolve their products from structure, never from the docking values.
    duplicates = frame[frame.duplicated(keys, keep=False)]
    alternatives = {tuple(key): g.index.tolist() for key, g in duplicates.groupby(keys)}
    ambiguous = np.zeros(n, bool)
    stereo_fallbacks = 0
    for (rid, position, sid), choices in alternatives.items():
        ambiguous |= (products.reaction_id.values == rid) & (products[f"synthon{position}_id"].to_numpy(dtype=np.int64, na_value=-1) == sid)
    if ambiguous.any():
        ambiguous_ids = np.flatnonzero(ambiguous)
        smiles = read_smiles_rows(table, ambiguous_ids)
        schemas = pd.read_csv(args.data / "spaces/reactions.tsv", sep="\t", dtype={"reaction_id": str}).set_index("reaction_id").Reaction.to_dict()
        for row, smiles_value in zip(ambiguous_ids, smiles):
            rid = str(products.reaction_id.iloc[row])
            wanted = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_value), isomericSmiles=True)
            choices = []
            for j in range(3):
                current = int(tuples[row, j])
                if current >= 0:
                    choices.append(alternatives.get((rid, j+1, int(frame.synthon_id.iloc[current])), [current]))
            matches, connectivity_matches = [], []
            wanted_connectivity = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_value), isomericSmiles=False)
            for candidate in itertools.product(*choices):
                assembled = assemble(frame.iloc[list(candidate)].smiles.tolist(), schemas[rid])
                if assembled == wanted:
                    matches.append(candidate)
                if Chem.MolToSmiles(Chem.MolFromSmiles(assembled), isomericSmiles=False) == wanted_connectivity:
                    connectivity_matches.append(candidate)
            if not matches and len(connectivity_matches) == 1:
                # Historical enumeration can differ in reaction stereoretention.
                # Use its product graph unchanged; only resolve the attachment
                # identity by connectivity (features/fingerprints are nonchiral).
                matches = connectivity_matches
                stereo_fallbacks += 1
            if len(matches) != 1:
                raise ValueError(f"Ambiguous product identity for row {row}: {len(matches)} matches")
            tuples[row, :len(matches[0])] = matches[0]
        print(f"Resolved {len(ambiguous_ids)} reused-ID products by exact structure", flush=True)
    reactions = sorted(slots)
    reaction_ids = pd.Categorical(products.reaction_id, categories=reactions).codes.astype(np.int16)
    enum_indices = np.zeros(n, np.int64)
    offset = 0
    layout = {}
    for rid, ss in slots.items():
        selected = products.reaction_id.values == rid
        idx = np.zeros(selected.sum(), np.int64)
        for j, axis in enumerate(ss):
            local = np.full(len(frame), -1, np.int32); local[axis] = np.arange(len(axis))
            idx = idx * len(axis) + local[tuples[selected, j]]
        enum_indices[selected] = offset + idx
        layout[rid] = {"offset": offset, "slots": [s.tolist() for s in ss]}
        offset += int(np.prod([len(s) for s in ss]))
    np.save(args.out / "tuples.npy", tuples)
    np.save(args.out / "reaction_ids.npy", reaction_ids)
    np.save(args.out / "enum_indices.npy", enum_indices)
    # Fixed-length IDs avoid pickled object arrays and support memory mapping.
    np.save(args.out / "product_ids.npy", products.product_id.to_numpy(dtype="S32"))
    if kind == "glide":
        np.save(args.out / "chemistry_filter.npy", products.passes_exp27_exact_filter.to_numpy(bool))
    frame.to_csv(args.out / "synthons.csv", index=False)
    print(f"Mapped {n:,} candidate tuples", flush=True)

    capped, failures = [], 0
    for smiles in frame.smiles:
        try:
            capped.append(cap_synthon(smiles))
        except (ValueError, RuntimeError):
            capped.append(smiles); failures += 1
    features = RDKitTeacher().encode(capped)
    np.save(args.out / "features.npy", features)

    fpdir = args.fingerprints / f"synthon_space_{args.scale}_enum_fps"
    metadata = json.loads((fpdir / "metadata.json").read_text())
    assert metadata["fp_type"] == "rdkit" and metadata["fp_size"] == 2048
    assert metadata["total_smiles"] == offset and metadata["total_invalid"] == 0
    fingerprints = None if args.skip_fingerprints else np.lib.format.open_memmap(args.out / "fingerprints.npy", mode="w+", dtype=np.uint8, shape=(n, 256))
    for chunk in range(0 if args.skip_fingerprints else metadata["n_chunks"]):
        description = json.load(gzip.open(fpdir / f"fps_{chunk:04d}.json.gz", "rt"))
        start = description["chunk_start_index"]
        with gzip.open(fpdir / f"fps_{chunk:04d}.npy.gz", "rb") as f:
            fp = np.load(f, allow_pickle=False)
        selected = (enum_indices >= start) & (enum_indices < start + len(fp))
        fingerprints[selected] = fp[enum_indices[selected] - start]
        if chunk % 10 == 0:
            print(f"Fingerprints {chunk+1}/{metadata['n_chunks']}", flush=True)
    if fingerprints is not None:
        fingerprints.flush()
    # Independent graph fingerprint audit detects any order or bit-layout error.
    sample = np.random.default_rng(3401).choice(n, min(64, n), replace=False) if not args.skip_fingerprints else []
    checked = []
    for smiles in read_smiles_rows(table, sample):
        bits = np.zeros(2048, np.uint8)
        DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), fpSize=2048), bits)
        checked.append(bits)
    orders = [None] if args.skip_fingerprints else []
    if not args.skip_fingerprints:
        for order in ("little", "big"):
            if all(np.array_equal(np.unpackbits(fingerprints[i], bitorder=order), bits) for i, bits in zip(sample, checked)):
                orders.append(order)
    if len(orders) != 1:
        raise ValueError(f"Fingerprint alignment audit failed: {orders}")
    for target in ("kif11", "pyrd", "tgfr1"):
        cols = [f"{target}_glide_le", f"{target}_glide_raw"] if kind == "glide" else [f"{target}_predicted_dock_score"]
        truth = pq.read_table(table, columns=cols).to_pandas().to_numpy(np.float32)
        # Kept in dedicated oracle file; the optimizer receives only queried rows.
        np.save(args.out / f"oracle_{target}.npy", truth)
    manifest = dict(scale=args.scale, kind=kind, products=n, synthon_rows=len(frame), reactions=reactions,
                    layout=layout, fingerprint_type="RDKit path 2048", bitorder=orders[0],
                    feature_type="capped synthon Morgan counts512 + four descriptors", uncappable_rows=failures,
                    score_table=str(table), space_sha256=hashlib.sha256(space_path.read_bytes()).hexdigest(),
                    alignment_audit_molecules=len(sample), ambiguous_products_resolved=int(ambiguous.sum()),
                    product_fingerprints_prepared=not args.skip_fingerprints,
                    ambiguous_stereo_connectivity_fallbacks=stereo_fallbacks,
                    preparation_seconds=time.perf_counter()-started)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    audit_product_identity(args.out)
    print(f"Prepared in {manifest['preparation_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
