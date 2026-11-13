"""The bundled SynthonOR example space, connector capping and RDKit assembly."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

DATA = Path(__file__).resolve().parent / "data"
CONNECTORS = {92, 93, 94}


def cap_synthon(smiles):
    """Cap U/Np/Pu connector bonds with H; reject graphs that cannot sanitize."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid synthon: {smiles}")
    rw = Chem.RWMol(mol)
    remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() in CONNECTORS]
    for index in remove:
        atom = rw.GetAtomWithIdx(index)
        if atom.GetDegree() != 1:
            raise ValueError(f"Connector must have one neighbor: {smiles}")
        neighbor = atom.GetNeighbors()[0]
        # Preserve bracketed atoms and aromatic [nH] when removing a connector.
        neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
    for index in sorted(remove, reverse=True):
        rw.RemoveAtom(index)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    result = Chem.MolToSmiles(capped, isomericSmiles=True)
    if Chem.MolFromSmiles(result) is None:
        raise ValueError(f"Capped synthon did not round-trip: {smiles}")
    return result


@lru_cache(maxsize=128)
def reaction(smarts):
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    return rxn


def assemble(smiles, smarts):
    candidates = set()
    for product_set in reaction(smarts).RunReactants(
        tuple(Chem.MolFromSmiles(s) for s in smiles)
    ):
        # These schemas assemble a single product, with connector atoms removed.
        if len(product_set) != 1:
            continue
        mol = product_set[0]
        try:
            Chem.SanitizeMol(mol)
            if any(a.GetAtomicNum() in CONNECTORS for a in mol.GetAtoms()):
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if Chem.MolFromSmiles(canonical) is not None:
                candidates.add(canonical)
        except (ValueError, RuntimeError):
            continue
    if not candidates:
        raise ValueError("No sanitizable assembled product")
    return min(candidates)


def slot_groups(frame):
    result = {}
    for rid, group in frame.groupby("reaction_id", sort=True):
        positions = sorted(group.position.unique())
        if positions not in ([1, 2], [1, 2, 3]):
            raise ValueError(f"{rid}: expected slots 1,2 or 1,2,3; got {positions}")
        result[str(rid)] = [
            group.index[group.position == p].to_numpy() for p in positions
        ]
    return result


def sample_products(frame, schemas, count, seed):
    """Uniform over reactions, then synthons. Deduplicate by canonical product."""
    groups = slot_groups(frame)
    rng = np.random.default_rng(seed)
    reactions = sorted(groups)
    records, seen = [], set()
    failed = 0
    for _ in range(count * 30):
        rid = reactions[int(rng.integers(len(reactions)))]
        rows = [int(rng.choice(ids)) for ids in groups[rid]]
        try:
            product = assemble(frame.iloc[rows].smiles.tolist(), schemas[rid])
        except (ValueError, RuntimeError):
            failed += 1
            continue
        if product in seen:
            continue
        seen.add(product)
        records.append(
            {
                "reaction_id": rid,
                "row_1": rows[0],
                "row_2": rows[1],
                "row_3": rows[2] if len(rows) == 3 else -1,
                "smiles": product,
            }
        )
        if len(records) == count:
            return pd.DataFrame(records), failed
    raise ValueError(
        f"Only sampled {len(records)} unique valid products; request fewer"
    )
