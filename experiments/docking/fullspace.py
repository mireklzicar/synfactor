"""Label-free synthon preparation and route identities for full Freedom grids."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from chemistry import cap_synthon
from teachers import RDKitTeacher


def prepare(library, out):
    library, out = Path(library), Path(out)
    source_hashes = {name: hashlib.sha256((library/name).read_bytes()).hexdigest()
                     for name in ("reactions.txt", "syntons.txt")}
    if (out/"manifest.json").exists():
        manifest = json.loads((out/"manifest.json").read_text())
        if manifest["source_sha256"] != source_hashes:
            raise ValueError("Full-space cache source changed")
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reactions = pd.read_csv(library/"reactions.txt", sep="\t", dtype={"reaction_id": str})
    frame = pd.read_csv(library/"syntons.txt", sep="\t", dtype=str).rename(
        columns={"SMILES": "smiles", "synton_id": "synthon_id", "synton#": "position"})
    frame.position = frame.position.astype(int)
    if frame.duplicated(["reaction_id", "position", "synthon_id"]).any():
        raise ValueError("Full release has ambiguous synthon identity")
    frame = frame.sort_values(["reaction_id", "position", "synthon_id"], kind="stable").reset_index(drop=True)
    names = sorted(reactions.reaction_id.tolist())
    if set(names) != set(frame.reaction_id):
        raise ValueError("Reaction and synthon files disagree")
    grids, total = [], 0
    for index, rid in enumerate(names):
        part = frame[frame.reaction_id == rid]
        slots = []
        arity = int(reactions.set_index("reaction_id").loc[rid, "components"])
        if arity not in (2, 3):
            raise ValueError("CP scorer currently supports two/three slots")
        for slot in range(1, arity+1):
            ids = part.index[part.position == slot].to_numpy()
            if not len(ids) or not np.array_equal(ids, np.arange(ids[0], ids[-1]+1)):
                raise ValueError("Missing or noncontiguous slot")
            slots.append([int(ids[0]), int(ids[-1]+1)])
        count = math.prod(b-a for a, b in slots)
        total += count
        grids.append(dict(reaction_index=index, reaction_id=rid, slots=slots, products=count))
    frame.to_parquet(out/"synthons.parquet", index=False)
    unique, inverse = np.unique(frame.smiles.values.astype(str), return_inverse=True)
    print(f"Preparing {len(frame):,} synthon rows; {len(unique):,} unique source structures", flush=True)
    # Encode unique structures once, then expand to typed synthon rows on disk.
    features = np.empty((len(unique), 516), np.float32)
    teacher = RDKitTeacher()
    capping_fallbacks = []
    for start in range(0, len(unique), 4096):
        capped = []
        for i, smi in enumerate(unique[start:start+4096], start):
            try:
                capped.append(cap_synthon(smi))
            except (ValueError, RuntimeError):
                # Retain the valid source synthon rather than dropping routes.
                # This is the same explicit fallback used by the subset cache.
                capped.append(smi); capping_fallbacks.append(i)
        features[start:start+len(capped)] = teacher.encode(capped)
        if start % (4096*20) == 0:
            print(f"Encoded {min(start+4096,len(unique)):,}/{len(unique):,}", flush=True)
    destination = np.lib.format.open_memmap(out/"features.npy", mode="w+", dtype=np.float32,
                                          shape=(len(frame), 516))
    for start in range(0, len(frame), 100000):
        destination[start:start+100000] = features[inverse[start:start+100000]]
    destination.flush()
    manifest = dict(source=str(library.resolve()), source_sha256=source_hashes,
                    products=total, synthon_rows=len(frame), reactions=names, grids=grids,
                    unique_source_structures=len(unique), capping_fallback_count=len(capping_fallbacks),
                    capping_fallback_policy="encode unchanged source synthon if capping fails",
                    features="capped Morgan count512 + four RDKit descriptors; no docking labels",
                    preparation_seconds=time.perf_counter()-started)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Prepared {total:,} routes in {manifest['preparation_seconds']:.1f}s", flush=True)
    return manifest


class FullSpace:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.manifest = json.loads((self.cache/"manifest.json").read_text())
        self.frame = pd.read_parquet(self.cache/"synthons.parquet")
        self.features = np.load(self.cache/"features.npy", mmap_mode="r")
        self.reactions = self.manifest["reactions"]
        self.grids = [dict(reaction_index=g["reaction_index"], slots=[np.arange(a,b,dtype=np.int64) for a,b in g["slots"]])
                      for g in self.manifest["grids"]]
        self.bounds = self.manifest["grids"]
        self.cumulative = np.cumsum([g["products"] for g in self.bounds], dtype=np.int64)
        self.lookup = pd.MultiIndex.from_frame(self.frame[["reaction_id", "position", "synthon_id"]])
        self.rid_lookup = {rid:i for i,rid in enumerate(self.reactions)}
        self.sids = self.frame.synthon_id.to_numpy()

    def decode(self, product_ids):
        tuples = np.full((len(product_ids),3), -1, np.int64)
        rids = np.empty(len(product_ids), np.int64)
        for i, pid in enumerate(product_ids):
            rid, *sids = pid.split("____")
            if rid not in self.rid_lookup or len(sids) != len(self.bounds[self.rid_lookup[rid]]["slots"]):
                raise ValueError(f"Unknown or malformed route {pid}")
            keys = [(rid,j+1,sid) for j,sid in enumerate(sids)]
            ids = self.lookup.get_indexer(keys)
            if np.any(ids < 0):
                raise ValueError(f"Route references missing synthons: {pid}")
            tuples[i,:len(ids)] = ids
            rids[i] = self.rid_lookup[rid]
        return tuples, rids

    def ids(self, tuples, rids):
        return ["____".join([self.reactions[int(rid)]]+[str(self.sids[j]) for j in row if j >= 0])
                for row,rid in zip(tuples,rids)]

    def sample(self, count, rng, excluded=()):
        seen = set(excluded)
        tuples, rids = [], []
        while len(tuples) < count:
            rid = int(np.searchsorted(self.cumulative, rng.integers(self.cumulative[-1]), side="right"))
            row = [int(rng.integers(a,b)) for a,b in self.bounds[rid]["slots"]]
            row += [-1]*(3-len(row))
            key = (rid,*row)
            if key not in seen:
                tuples.append(row);rids.append(rid);seen.add(key)
        return np.asarray(tuples,np.int64), np.asarray(rids,np.int64)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();prepare(a.library,a.out)
