"""Local CPU-only, serial refinement of completed and initially evaluated cells.

Never retries a failed/interrupted refinement or overwrites existing results.
No oracle, SSH, cloud or campaign-control operations are performed.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from experiments.docking.live_monitor import (evaluation_complete, now, process_identity,
    process_state, read, sha, write)

METHODS = ('synfactor', 'ga_dcso_v14_screening', 'spacehasten_fast', 'spacehasten_accurate')


def verified_result(output, molecules):
    result = read(output)
    if result.get('input_sha256') != sha(molecules):
        raise ValueError('Existing refinement input hash differs from current molecular CSV')
    if len(set(result.get('selected_product_ids', []))) != 100:
        raise ValueError('Refinement must contain100 distinct selected identities')
    if result['upper_bound'] < result['value']-1e-7:
        raise ValueError('Invalid refinement utility interval')
    if not result.get('source_sha256'):
        raise ValueError('Missing refinement source provenance')
    for digest in result['source_sha256'].values():
        snapshot = output.parent/'sources'/(digest+'.py')
        if not snapshot.exists() or sha(snapshot) != digest:
            raise ValueError('Missing or changed refinement source snapshot')
    return {k:result[k] for k in ('value','upper_bound','certified','status','wall_seconds','input_sha256')}


def initial_pending(cells, monitor):
    for cell in cells:
        if (cell/'evaluation_live/comparison.csv').exists():
            continue
        job = monitor.get('evaluations', {}).get(cell.name)
        if job is None or (job.get('status') == 'running'
                           and process_state(job['process']) == 'running'):
            return True
    return False


def progress(campaign, state, monitor):
    records = []
    for key, job in sorted(state['jobs'].items()):
        run, method = key.split('/')
        cell = campaign/run
        record = dict(run=run, method=method, job_status=job['status'], **job.get('result', {}))
        if 'error' in job:
            record['error'] = job['error']
        summary_path = cell/'evaluation_live/summary.json'
        if summary_path.exists():
            summary = read(summary_path)
            record.update(target=summary['target'], seed=summary['seed'])
            match = next((c for c in summary['cells'] if c['method'] == method), None)
            if match:
                record['initial_druglike_opdiv'] = next(p for p in match['domains']['druglike-v1']['portfolios']
                                                        if p['metric'] == 'OPDiv')
        records.append(record)
    payload = dict(updated_utc=now(), watcher_status=state['status'],
                   initial_evaluation_failures={k:v for k,v in monitor.get('evaluations',{}).items()
                                                if v.get('status')=='failed'},
                   campaign_process_state=process_state(monitor['process']), records=records)
    write(campaign/'refined_progress.json', payload)
    lines = ['# Refined portfolio progress', '', 'Mean negative docking energy; higher is better. Bounds refer to the complete eligible molecular archive.', '',
             '| Target | Method | All-valid OPDiv interval | Certified | Druglike OPDiv interval | Status |',
             '|---|---|---|---|---|---|']
    def interval(lower, upper):
        return '—' if lower is None or upper is None else f'[{lower:.5f}, {upper:.5f}]'
    for r in records:
        drug = r.get('initial_druglike_opdiv', {})
        lines.append(f"| {r.get('target',r['run'])} | {r['method']} | {interval(r.get('value'),r.get('upper_bound'))} | {r.get('certified','—')} | {interval(drug.get('feasible_lower_bound'),drug.get('upper_bound'))} | {r['job_status']} |")
    temp = campaign/'refined_progress.md.tmp'; temp.write_text('\n'.join(lines)+'\n')
    temp.replace(campaign/'refined_progress.md')


def tick(args, state):
    """Process at most one local job. Returns True when no further work is expected."""
    monitor = read(args.campaign/'monitor_state.json')
    cells = sorted(p.parent for p in args.campaign.glob('sf_*/CELL_DONE'))
    # Recover finished children, or wait for an existing child after supervisor restart.
    active = False
    for key, job in state['jobs'].items():
        if job['status'] != 'running':
            continue
        if job.get('process') and process_state(job['process']) == 'running':
            active = True; continue
        run, method = key.split('/')
        cell = args.campaign/run
        try:
            molecules = cell/'evaluation_live'/f'{method}__all-valid__molecules.csv.gz'
            if sha(molecules) != job['input_sha256']:
                raise ValueError('Molecular CSV changed after refinement launch')
            job['result'] = verified_result(cell/'refinement'/f'{method}__all-valid.json',
                molecules)
            job.update(status='complete', completed_utc=now())
        except Exception as error:
            job.update(status='failed_no_retry', error=f'{type(error).__name__}: {error}')
    for cell in cells:
        evaluation = cell/'evaluation_live'
        if not (evaluation/'comparison.csv').exists():
            continue
        config = read(cell/'config.json')
        if not evaluation_complete(evaluation, sha(cell/'state.json'), config):
            raise ValueError(f'Initial evaluation is not identity-bound and complete: {cell.name}')
        for method in METHODS:
            key = cell.name+'/'+method
            output = cell/'refinement'/f'{method}__all-valid.json'
            molecules = evaluation/f'{method}__all-valid__molecules.csv.gz'
            if key in state['jobs']:
                # Revalidate already completed outputs; failed jobs are terminal.
                if state['jobs'][key]['status'] == 'complete':
                    state['jobs'][key]['result'] = verified_result(output, molecules)
                continue
            if output.exists():
                try:
                    result = verified_result(output, molecules)
                    state['jobs'][key] = dict(status='complete', result=result, recovered_utc=now())
                except Exception as error:
                    state['jobs'][key] = dict(status='failed_no_retry', error=f'{type(error).__name__}: {error}')
                continue
            if active:
                continue
            state['jobs'][key] = dict(status='running', started_utc=now(), input_sha256=sha(molecules))
            write(args.campaign/'refinement_watch_state.json', state)
            output.parent.mkdir(exist_ok=True)
            try:
                command = [str(args.eval_python), '-m', 'experiments.docking.live_refine',
                           '--molecules', str(molecules), '--out', str(output),
                           '--benchmark-repo', str(args.benchmark_repo)]
                with (output.parent/f'{method}__all-valid.log').open('a') as log:
                    child = subprocess.Popen(command, cwd=args.frozen_root, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
                state['jobs'][key]['process'] = process_identity(child.pid)
                active = True
            except Exception as error:
                state['jobs'][key].update(status='failed_no_retry', error=f'{type(error).__name__}: {error}')
            write(args.campaign/'refinement_watch_state.json', state)
    terminal = all(cell.name+'/'+method in state['jobs'] and
                   state['jobs'][cell.name+'/'+method]['status'] != 'running' for cell in cells for method in METHODS)
    done = not active and ((len(cells) >= args.expected_targets and terminal)
                          or (process_state(monitor['process']) != 'running' and not initial_pending(cells, monitor) and terminal))
    # Failed initial evaluations have no molecular inputs; they must not keep the watcher alive forever.
    if not active and process_state(monitor['process']) != 'running' and not initial_pending(cells, monitor):
        ready = [c for c in cells if (c/'evaluation_live/comparison.csv').exists()]
        done = all(c.name+'/'+m in state['jobs'] and state['jobs'][c.name+'/'+m]['status'] != 'running'
                   for c in ready for m in METHODS)
    state['status'] = 'finished' if done else 'watching'
    write(args.campaign/'refinement_watch_state.json', state)
    progress(args.campaign, state, monitor)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--eval-python', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, default=Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2'))
    parser.add_argument('--expected-targets', type=int, default=12)
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60 or args.expected_targets < 1:
        parser.error('Need5–60s polling and a positive target count')
    for name in ('campaign','frozen_root','benchmark_repo'):
        setattr(args,name,getattr(args,name).resolve())
    args.eval_python = args.eval_python.absolute()
    for name in ('live_refine.py','live_evaluate.py','live_monitor.py','live_refine_watch.py'):
        if not (args.frozen_root/'experiments/docking'/name).is_file():
            parser.error(f'Missing frozen source: {name}')
    sources = list((args.frozen_root/'experiments/docking').glob('*.py')) + [
        args.benchmark_repo/'scripts/analyze_diversity_portfolios.py', args.benchmark_repo/'scripts/refine_diversity_bounds.py']
    binding = dict(campaign=str(args.campaign), frozen_root=str(args.frozen_root), eval_python=str(args.eval_python),
                   expected_targets=args.expected_targets, source_sha256={str(p):sha(p) for p in sources})
    path = args.campaign/'refinement_watch_state.json'
    with (args.campaign/'refinement_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = read(path) if path.exists() else dict(binding=binding, created_utc=now(), jobs={}, status='watching')
        if state['binding'] != binding:
            raise ValueError('Refinement watcher source/configuration changed; refusing unsafe resume')
        while True:
            if any(sha(p) != digest for p,digest in binding['source_sha256'].items()):
                raise RuntimeError('Frozen refinement source changed')
            try:
                if tick(args,state):
                    return
            except Exception as error:
                state.update(status='failed', error=f'{type(error).__name__}: {error}', failed_utc=now())
                write(path,state)
                raise
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
