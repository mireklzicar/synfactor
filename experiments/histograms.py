"""Audit additive descriptor histograms; NOT a native SpaceProp benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem

from chemistry import DATA, CONNECTORS, reaction, slot_groups
from .baselines import integer_histogram, shi_predict
from .compare import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/comparison"))
    args = parser.parse_args()
    products = pd.read_csv(args.out / "products.csv", dtype={"reaction_id": str})
    synthons = pd.read_csv(args.out / "synthons.csv", dtype={"reaction_id": str})
    data = np.load(args.out / "rdkit.npz")
    bp = np.load(args.out / "basis.npz")
    y, names = data["y"], data["names"].tolist()
    pred = shi_predict(y, bp["basis"], bp["refs"])
    result, histograms = [], {}
    for rid, subset in products.groupby("reaction_id", sort=True):
        ii = subset.index.values
        ref = int(bp["refs"][ii[0]])
        slots = synthons[synthons.reaction_id == rid]
        for j, name in enumerate(names):
            error = np.abs(y[ii, j] - pred[ii, j])
            record = dict(reaction_id=rid, endpoint=name, max_abs_error=float(error.max()),
                          fraction_exact=float(np.mean(error < 1e-8)), tuples=len(ii))
            if name in ("RDKit/HAC", "RDKit/HBA", "RDKit/HBD"):
                values = []
                for slot, sg in slots.groupby("position", sort=True):
                    basis_rows = [next(i for i in ii if int(products.loc[i, f"local_{slot}"]) == int(local)
                                       and all(int(products.loc[i, f"local_{k}"]) <= 0 for k in (1, 2, 3) if k != slot))
                                  for local in sg.local_index]
                    values.append(np.rint(y[basis_rows, j] - y[ref, j]).astype(int))
                start = time.perf_counter()
                counts = integer_histogram(values, int(round(y[ref, j])))
                elapsed = time.perf_counter() - start
                exact = Counter(int(v) for v in y[ii, j])
                keys = counts.keys() | exact.keys()
                tv = sum(abs(counts[k] - exact[k]) for k in keys) / (2 * len(ii))
                assert sum(counts.values()) == len(ii)
                record.update(histogram_total_variation=tv, histogram_exact=counts == exact, convolution_seconds=elapsed)
                histograms[f"{rid}/{name}"] = {"additive": dict(counts), "enumerated": dict(exact)}
            result.append(record)
    pd.DataFrame(result).to_csv(args.out / "descriptor_additivity.csv", index=False)
    dump(args.out / "descriptor_histograms.json", histograms)

    # An exact HAC control on supported bundled schemas, without enumeration.
    # Establish atom conservation from the actual reaction SMARTS first.
    frame = pd.read_csv(DATA / "synthon_space_1M.tsv", sep="\t", dtype={"reaction_id": str})
    schemas = pd.read_csv(DATA / "reactions.tsv", sep="\t", dtype={"reaction_id": str}).set_index("reaction_id").Reaction.to_dict()
    start = time.perf_counter()
    hac = np.array([sum(a.GetAtomicNum() > 1 and a.GetAtomicNum() not in CONNECTORS
                        for a in Chem.MolFromSmiles(s).GetAtoms()) for s in frame.smiles])
    feature_time = time.perf_counter() - start
    all_counts, reaction_info, excluded = Counter(), [], []
    start = time.perf_counter()
    for rid, slots in slot_groups(frame).items():
        rxn = reaction(schemas[rid])
        reactant_atoms = [a for mol in rxn.GetReactants() for a in mol.GetAtoms()]
        product_atoms = [a for mol in rxn.GetProducts() for a in mol.GetAtoms()]
        r_maps = sorted(a.GetAtomMapNum() for a in reactant_atoms if a.GetAtomMapNum())
        p_maps = sorted(a.GetAtomMapNum() for a in product_atoms if a.GetAtomMapNum())
        if (r_maps != p_maps or len(set(r_maps)) != len(r_maps)
                or any(not a.GetAtomMapNum() and a.GetAtomicNum() not in CONNECTORS for a in reactant_atoms)
                or any(not a.GetAtomMapNum() for a in product_atoms)):
            excluded.append(rid)
            continue
        counts = integer_histogram([hac[ids] for ids in slots])
        expected = int(np.prod([len(ids) for ids in slots]))
        assert sum(counts.values()) == expected
        all_counts.update(counts)
        reaction_info.append({"reaction_id": rid, "tuples": expected})
    elapsed = time.perf_counter() - start
    dump(args.out / "full_space_hac.json", {
        "method": "exact additive HAC convolution control, not native SpaceProp",
        "counting_unit": "syntactic synthon combinations; not deduplicated molecules; invalid assemblies not checked",
        "synthon_rows": len(frame), "reactions": len(reaction_info), "tuple_count": sum(all_counts.values()),
        "excluded_reactions_not_proved_atom_conserving": excluded,
        "histogram": dict(sorted(all_counts.items())), "reaction_counts": reaction_info,
        "synthon_feature_seconds": feature_time, "schema_validation_and_convolution_seconds": elapsed,
        "condition": "HAC counts assume successful assembly; atom conservation checked for every schema",
    })
    print(json.dumps({"tuple_count": sum(all_counts.values()), "feature_seconds": feature_time, "convolution_seconds": elapsed}))


if __name__ == "__main__":
    main()
