"""Read-only feasibility audit; writes aggregate evidence only, never solves/docks."""
from pathlib import Path
import hashlib,json,math
from datetime import datetime,timezone
import pandas as pd
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'runs/docking_live_campaign/suite_info20/sf_freedom_seed0_tyk2_info20'
PUBLIC_OLD=ROOT/'benchmarks/docking_fullspace/tyk2_seed0/all_valid_refinement.json'
PRIVATE=OLD/'refinement/spacehasten_accurate__all-valid.json'
OLD_DOMAIN=OLD/'evaluation_live/spacehasten_accurate__all-valid__molecules.csv.gz'
METHOD='spacehasten_accurate'
OUT=ROOT/'benchmarks/docking_fullspace/tyk2_v2/spacehasten_accurate_incumbent_verification.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(smiles):
    mol=Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol,isomericSmiles=True)
def domain(path):
    rows=pd.read_csv(path).to_dict('records')
    for row in rows:
        row['_canonical']=canonical(row['smiles'])
        assert math.isfinite(row['score'])
    assert len({r['_canonical'] for r in rows})==len(rows)
    assert len({r['product_id'] for r in rows})==len(rows)
    return rows

def semantic_digest(rows):
    values=sorted((r['_canonical'],format(r['score'],'.12f')) for r in rows)
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()

prior=json.loads(PRIVATE.read_text())
public=json.loads(PUBLIC_OLD.read_text())[METHOD]
assert prior['rdkit']==rdBase.rdkitVersion
assert digest(OLD_DOMAIN)==prior['input_sha256']==public['input_sha256']
assert abs(prior['value']-public['value'])<1e-12
ids=prior['selected_product_ids']
assert len(ids)==len(set(ids))==100
old=domain(OLD_DOMAIN);old_by_id={r['product_id']:r for r in old}
old_selected=[old_by_id[pid] for pid in ids]
assert len({r['_canonical'] for r in old_selected})==100
assert abs(-sum(r['score'] for r in old_selected)/100-prior['value'])<1e-12
old_summary=json.loads((OLD/'evaluation_live/summary.json').read_text())
generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
current_proofs=[]
best_upper=None
for policy in ('raw','portfolio'):
    folder=ROOT/f'runs/tyk2_v2/cells/tyk2_{policy}_uniform_s0/evaluation_10k'
    summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text())
    assert summary['charged_calls']==10000 and summary['comparison_verified'] is True
    for key in ('target','seed','grid_sha256','warm_sha256'):
        assert summary[key]==old_summary[key]
    assert summary['rdkit']==rdBase.rdkitVersion
    path=folder/'spacehasten_accurate__all_valid.csv.gz'
    rows=domain(path);by_canonical={r['_canonical']:r for r in rows}
    assert len(rows)==len(old)==9965
    assert semantic_digest(rows)==semantic_digest(old)
    selected=[by_canonical[r['_canonical']] for r in old_selected]
    assert len({r['product_id'] for r in selected})==100
    same_ids=sum(a['product_id']==b['product_id'] for a,b in zip(old_selected,selected))
    same_scores=sum(abs(a['score']-b['score'])<1e-12 for a,b in zip(old_selected,selected))
    assert same_ids==same_scores==100
    fingerprints=[generator.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in selected]
    values=[value for i,fp in enumerate(fingerprints) for value in DataStructs.BulkTanimotoSimilarity(fp,fingerprints[:i])]
    assert len(values)==4950 and max(values)<.30
    incumbent=-sum(r['score'] for r in selected)/100
    assert abs(incumbent-prior['value'])<1e-12
    cell=next(c for c in summary['cells'] if c['method']==METHOD)
    solver=next(p for p in cell['portfolios'] if p['metric']=='OPDiv')
    assert solver['full_domain'] is True and solver['selected_count']==100
    upper=float(solver['upper_bound'])
    assert incumbent<=upper+1e-12
    best_upper=upper if best_upper is None else min(best_upper,upper)
    current_proofs.append(dict(reference_policy=policy,current_domain_sha256=digest(path),
        current_evaluation_sha256=digest(summary_path),canonical_domain_semantic_sha256=semantic_digest(rows),
        domain_molecules=len(rows),incumbent_members_found=100,unchanged_product_identity_count=same_ids,
        unchanged_score_count=same_scores,pairwise_comparisons=len(values),maximum_tanimoto=max(values),
        strict_threshold=.30,all_pairwise_tanimoto_strictly_below_threshold=True,
        recomputed_raw_mean=-incumbent,recomputed_feasible_utility=incumbent,
        recorded_solver_feasible_lower_bound=solver['feasible_lower_bound'],
        recorded_solver_global_upper_bound=upper,recorded_solver_status=solver['status'],
        recorded_solver_certified=solver['certified']))

report=dict(schema_version=1,verified_utc=datetime.now(timezone.utc).isoformat(),
    method=METHOD,target='tyk2',seed=0,charged_calls=10000,eligibility='all valid; canonical isomeric molecules; no druglike filter',
    contract='100 distinct molecules; Morgan radius2/2048/no chirality; every pair Tanimoto strictly <0.30',
    rdkit=rdBase.rdkitVersion,verification='prior feasible incumbent remains feasible in both current full domains',
    source_sha256=dict(prior_public_aggregate=digest(PUBLIC_OLD),prior_private_incumbent=digest(PRIVATE),
        prior_domain=digest(OLD_DOMAIN),prior_evaluation=digest(OLD/'evaluation_live/summary.json'),
        verification_script=digest(Path(__file__))),
    prior_private_incumbent_count=100,verified_feasible_lower_bound=incumbent,
    strongest_current_full_domain_upper_bound=best_upper,best_known_gap=best_upper-incumbent,
    best_known_certified_optimal=False,current_domain_checks=current_proofs,
    reporting='Post-hoc verified known-incumbent lower bound; retain original solver-run values separately. The upper bound comes only from current full-domain solves.',
    restricted_support_optimality_reused_as_global=False,
    original_solver_results_modified=False,new_optimization_calls=0,new_oracle_calls=0,
    raw_structures_or_product_ids_exported=False)
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(ROOT/'runs/tyk2_v2/audit/spacehasten_accurate_incumbent_verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:report[key] for key in ('verified_feasible_lower_bound','strongest_current_full_domain_upper_bound','best_known_gap','best_known_certified_optimal')}))
print('maximum_tanimoto',current_proofs[0]['maximum_tanimoto'])
print('artifact',OUT)
