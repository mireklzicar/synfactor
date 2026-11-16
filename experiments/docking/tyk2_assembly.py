"""Label-free native assembly and true product fingerprints for bounded pools.

The prepared index must be a trusted local SynthonAssembler pickle. No docking
scores are read. Existing library assets are never modified. Cache identity
includes native sources, the index, and RDKit; mismatched caches fail closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class AssemblyCache:
    def __init__(self, library, cache, assembler_source=None):
        started = time.perf_counter()
        if assembler_source is not None:
            sys.path.insert(0, str(Path(assembler_source).resolve()))
        from synthon_assembler import assembly, index
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdFingerprintGenerator
        self.Chem = Chem
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=False)
        self.library, self.cache = Path(library).resolve(), Path(cache).resolve()
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.identity = dict(
            index_sha256=digest(self.library / 'synthon_index.pkl'),
            reactions_sha256=digest(self.library / 'reactions.txt'),
            synthons_sha256=digest(self.library / 'syntons.txt'),
            assembly_sha256=digest(assembly.__file__), index_source_sha256=digest(index.__file__),
            rdkit=rdBase.rdkitVersion, product_prep='minimal',
            canonical_isomeric_smiles=True, fingerprint='Morgan-radius2-2048-no-chirality')
        self.connection = sqlite3.connect(self.cache)
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS molecules (product_id TEXT PRIMARY KEY,smiles TEXT,status TEXT NOT NULL,error TEXT)')
        expected = json.dumps(self.identity, sort_keys=True)
        stored = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if stored is not None and stored[0] != expected:
            self.connection.close()
            raise ValueError('Assembly cache identity differs from the native library/source/RDKit')
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (expected,))
        self.connection.commit()
        blocker = rdBase.BlockLogs()
        try:
            self.index = index.SynthonIndex.load(self.library / 'synthon_index.pkl')
        finally:
            del blocker
        self._fingerprints = {}
        self.load_seconds = time.perf_counter() - started
        self.last_stats = {}

    def close(self):
        self.connection.close()

    def assemble(self, product_ids):
        """Return aligned rows, including explicit failures and repeated IDs."""
        started = time.perf_counter()
        ids = [str(pid) for pid in product_ids]
        unique = list(dict.fromkeys(ids))
        found = {}
        for start in range(0, len(unique), 500):
            subset = unique[start:start+500]
            placeholders = ','.join('?' for _ in subset)
            for row in self.connection.execute(
                    f'SELECT product_id,smiles,status,error FROM molecules WHERE product_id IN ({placeholders})', subset):
                found[row[0]] = dict(zip(('product_id', 'smiles', 'status', 'error'), row))
        missing = [pid for pid in unique if pid not in found]
        for pid in missing:
            try:
                rid, *synthons = pid.split('____')
                if not rid or not synthons or any(not synthon for synthon in synthons):
                    raise ValueError('Malformed route identity')
                # Same native entry point and defaults as Navigator's production adapter.
                smiles = self.index.assemble_by_ids(rid, synthons)
                mol = self.Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError('Assembled product cannot be sanitized')
                smiles = self.Chem.MolToSmiles(mol, isomericSmiles=True)
                row = dict(product_id=pid, smiles=smiles, status='valid', error=None)
            except Exception as error:
                row = dict(product_id=pid, smiles=None, status='assembly_failed',
                           error=f'{type(error).__name__}: {error}')
            found[pid] = row
            self.connection.execute('INSERT INTO molecules VALUES (?,?,?,?)',
                                    tuple(row[k] for k in ('product_id','smiles','status','error')))
        self.connection.commit()
        result = [dict(found[pid]) for pid in ids]
        self.last_stats = dict(rows=len(ids), unique=len(unique), new_assemblies=len(missing),
                               cache_hits=len(unique)-len(missing),
                               valid=sum(row['status']=='valid' for row in result),
                               seconds=time.perf_counter()-started)
        return result

    def fingerprints(self, product_ids):
        """Aligned ExplicitBitVect/None; suitable for BulkTanimotoSimilarity."""
        rows = self.assemble(product_ids)
        for row in rows:
            pid = row['product_id']
            if pid not in self._fingerprints:
                self._fingerprints[pid] = (self.generator.GetFingerprint(self.Chem.MolFromSmiles(row['smiles']))
                                           if row['status']=='valid' else None)
        return [self._fingerprints[row['product_id']] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--assembler-source', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True, help='CSV with product_id and smiles; no labels used')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.candidates.open() as handle:
        for row in csv.DictReader(handle):
            rows.append({key: row[key] for key in ('product_id', 'smiles')})
            if len(rows) == args.limit:
                break
    cache = AssemblyCache(args.library, args.cache, args.assembler_source)
    assembled = cache.assemble([row['product_id'] for row in rows])
    assembly_stats = dict(cache.last_stats)
    mismatches = []
    for original, actual in zip(rows, assembled):
        mol = cache.Chem.MolFromSmiles(original['smiles'])
        expected = cache.Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None
        if actual['smiles'] != expected:
            mismatches.append(dict(product_id=original['product_id'], expected=expected, actual=actual['smiles']))
    started = time.perf_counter()
    fingerprints = cache.fingerprints([row['product_id'] for row in rows])
    fingerprint_seconds = time.perf_counter()-started
    report = dict(identity=cache.identity, candidates_sha256=digest(args.candidates),
                  rows=len(rows), load_seconds=cache.load_seconds, assembly=assembly_stats,
                  structure_mismatches=len(mismatches), mismatch_examples=mismatches[:20],
                  fingerprint_seconds=fingerprint_seconds,
                  valid_fingerprints=sum(fp is not None for fp in fingerprints),
                  docking_calls=0, label_columns_read=[])
    (args.out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.out/'assembled.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('product_id','smiles','status','error'))
        writer.writeheader();writer.writerows(assembled)
    cache.close()
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit('Native assembly differs from expected structures; qualification failed')


if __name__ == '__main__':
    main()
