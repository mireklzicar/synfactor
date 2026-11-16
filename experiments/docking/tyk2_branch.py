"""Prepare an optional, auditable development fork without any oracle calls.

Only the first three adaptive batches (4,000 charged calls including warmup) are
inherited by default. The new run still has a total 10,000-call budget. This is
an adaptive development fork, not an independent replicate. The driver creates
its own new config.json later; this utility never launches that driver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from .tyk2_evaluate import BENCHMARK, load_json, sha, verify_partial


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path):
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular evidence file: {path.name}')
    return path.read_bytes()


def prefix_digest(state, inherited_rounds):
    prefix=dict(history=state['history'][:1000*(inherited_rounds+1)],
                rounds=state['rounds'][:inherited_rounds])
    return hashlib.sha256(json.dumps(prefix,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_lineage(root, state, config):
    """A development fork inherits charged evidence, never a fresh free warmup."""
    path=root/'lineage.json'
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError('Lineage must be a regular immutable file')
    lineage=load_json(path)
    count=lineage.get('inherited_rounds')
    if (lineage.get('schema_version')!=1 or type(count) is not int or not 1<=count<=8
            or lineage.get('inherited_calls')!=1000*(count+1)
            or lineage.get('total_budget')!=10000 or lineage.get('remaining_calls')!=10000-1000*(count+1)
            or lineage.get('development_fork') is not True or lineage.get('independent_replicate') is not False):
        raise ValueError('Invalid development lineage or inherited budget')
    current_id=config.get('run_id',root.name)
    parent_id=lineage.get('parent_run_id')
    if (not parent_id or current_id==parent_id or lineage.get('destination_run_id')!=current_id
            or lineage.get('parent_was_branched') is not False):
        raise ValueError('Lineage run identity mismatch or unsupported nested fork')
    parent_path=root/'lineage_parent_config.json'
    if parent_path.is_symlink() or sha(parent_path)!=lineage.get('parent_config_sha256'):
        raise ValueError('Parent configuration lineage hash mismatch')
    parent=load_json(parent_path)
    if parent.get('run_id',parent_id)!=parent_id:
        raise ValueError('Parent run ID differs from lineage')
    if parent.get('policy')!='portfolio':
        raise ValueError('Lineage did not originate from the portfolio policy')
    for key in ('target','seed','budget','warm_charged','batch_size','grid_sha256','warm','oracle_protocol_sha256'):
        if parent.get(key)!=config.get(key):
            raise ValueError('Fork changed its inherited oracle/warm/budget contract')
    if (len(state['history'])<lineage['inherited_calls'] or len(state.get('rounds',[]))<count
            or prefix_digest(state,count)!=lineage.get('prefix_sha256')):
        raise ValueError('Inherited parent prefix was modified or truncated')
    required={f'batch_{i:02d}/{name}' for i in range(1,count+1)
              for name in ('proposal.json','normalized.json','search.json')}
    if set(lineage.get('batch_sha256',{}))!=required:
        raise ValueError('Lineage does not bind exactly the inherited evidence files')
    for relative in sorted(required):
        artifact=root/relative
        if artifact.is_symlink() or artifact.parent.is_symlink() or sha(artifact)!=lineage['batch_sha256'][relative]:
            raise ValueError('Inherited batch evidence hash mismatch')
    verify_run_bindings(root,config,count,lineage)
    return dict(lineage_sha256=sha(path),parent_run_id=parent_id,
                destination_run_id=current_id,inherited_rounds=count,
                inherited_calls=lineage['inherited_calls'],prefix_sha256=lineage['prefix_sha256'],
                parent_state_sha256=lineage['parent_state_sha256'],
                parent_config_sha256=lineage['parent_config_sha256'],
                branch_utility_sha256=lineage.get('branch_utility_sha256'),
                development_fork=True,independent_replicate=False)


def verify_run_bindings(root,config,rounds,lineage=None):
    """Inherited requests belong to parent; fresh requests belong to this fork."""
    for iteration in range(1,rounds+1):
        folder=root/f'batch_{iteration:02d}'
        expected=(lineage['parent_run_id'] if lineage is not None and iteration<=lineage['inherited_rounds']
                  else config.get('run_id',root.name))
        proposal,response=load_json(folder/'proposal.json'),load_json(folder/'normalized.json')
        if any(obj.get('run_id')!=expected or obj.get('batch_id')!=folder.name
               or obj.get('target')!=config['target'] or obj.get('seed')!=config['seed']
               for obj in (proposal,response)):
            raise ValueError('Inherited or new batch run/target/seed identity mismatch')


def create_branch(parent, destination, benchmark_repo=BENCHMARK, inherited_rounds=3):
    parent,destination=Path(parent).absolute(),Path(destination).absolute()
    if parent.is_symlink() or destination.is_symlink() or not parent.is_dir():
        raise ValueError('Parent and destination must not be symlinks')
    if parent.resolve()==destination.resolve() or parent.resolve() in destination.resolve().parents:
        raise ValueError('Fork must be outside its parent cell')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Destination must be new or empty')
    if not 1<=inherited_rounds<=8:
        raise ValueError('Inherit one through eight complete adaptive rounds')
    if (parent/'lineage.json').exists():
        raise ValueError('Nested forks are unsupported; no other branch labels may be imported')
    # Read the advancing parent state exactly once. Completed batch evidence is
    # copied into a private snapshot before validation; no later state reread.
    state_bytes=regular_bytes(parent/'state.json')
    config_bytes=regular_bytes(parent/'config.json')
    state,config=json.loads(state_bytes),json.loads(config_bytes)
    if config.get('policy')!='portfolio':
        raise ValueError('This development fork must originate from the portfolio policy')
    parent_run_id=config.get('run_id',parent.name)
    if parent_run_id==destination.name:
        raise ValueError('Fork requires a distinct run ID')
    inherited_calls=1000*(inherited_rounds+1)
    if len(state['history'])<inherited_calls or len(state.get('rounds',[]))<inherited_rounds:
        raise ValueError('Parent has not completed the requested inherited budget')
    prefix=dict(history=state['history'][:inherited_calls],rounds=state['rounds'][:inherited_rounds])
    for iteration,record in enumerate(prefix['rounds'],1):
        if record.get('iteration')!=iteration or record.get('calls')!=1000*(iteration+1):
            raise ValueError('Parent round ledger does not match inherited charged count')
    prefix_bytes=(json.dumps(prefix,indent=2,allow_nan=False)+'\n').encode()
    copies={'state.json':prefix_bytes,'lineage_parent_config.json':config_bytes,
            'remote_preflight.json':regular_bytes(parent/'remote_preflight.json')}
    batch_hashes={}
    for iteration in range(1,inherited_rounds+1):
        for name in ('proposal.json','normalized.json','search.json'):
            relative=f'batch_{iteration:02d}/{name}'
            copies[relative]=regular_bytes(parent/relative)
            batch_hashes[relative]=digest(copies[relative])
    with tempfile.TemporaryDirectory(prefix='synfactor-fork-verify-') as directory:
        snapshot=Path(directory)/parent_run_id;snapshot.mkdir()
        for relative,data in copies.items():
            path=snapshot/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (snapshot/'config.json').write_bytes(config_bytes)
        verified,_,proof=verify_partial(snapshot/'state.json',inherited_rounds,Path(benchmark_repo),
                                        config['target'],config['seed'])
        verify_run_bindings(snapshot,config,inherited_rounds)
        if len(verified)!=inherited_calls:
            raise ValueError('Verified inherited budget differs')
    lineage=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
                 parent_run_id=parent_run_id,destination_run_id=destination.name,
                 parent_config_sha256=digest(config_bytes),parent_state_sha256=digest(state_bytes),
                 parent_state_calls_at_snapshot=len(state['history']),parent_was_branched=False,
                 inherited_rounds=inherited_rounds,inherited_calls=inherited_calls,
                 total_budget=10000,remaining_calls=10000-inherited_calls,
                 prefix_sha256=prefix_digest(prefix,inherited_rounds),prefix_state_sha256=digest(prefix_bytes),
                 batch_sha256=batch_hashes,parent_preflight_sha256=digest(copies['remote_preflight.json']),
                 branch_utility_sha256=sha(Path(__file__)),
                 evaluator_source_sha256=sha(Path(__file__).with_name('tyk2_evaluate.py')),
                 verification=proof,development_fork=True,independent_replicate=False,
                 accounting='Inherited calls remain charged once within the child 10000-call budget',
                 oracle_calls_submitted=0)
    copies['lineage.json']=(json.dumps(lineage,indent=2,allow_nan=False)+'\n').encode()
    # Recheck after validation: never overwrite a destination created meanwhile.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination became nonempty during verification')
    destination.mkdir(parents=True,exist_ok=True)
    for relative,data in copies.items():
        path=destination/relative;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data)
    return lineage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inherited-rounds',type=int,default=3)
    parser.add_argument('--benchmark-repo',type=Path,default=BENCHMARK)
    args=parser.parse_args()
    lineage=create_branch(args.parent,args.out,args.benchmark_repo,args.inherited_rounds)
    print(json.dumps({key:lineage[key] for key in ('parent_run_id','destination_run_id','inherited_calls',
                      'remaining_calls','prefix_sha256','development_fork','oracle_calls_submitted')},indent=2))


if __name__=='__main__':
    main()
