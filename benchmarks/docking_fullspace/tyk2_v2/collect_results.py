"""Publish aggregate TYK2 development trajectories; never export molecules.

Run from the repository root with an environment containing pandas/matplotlib:
  python benchmarks/docking_fullspace/tyk2_v2/collect_results.py --runs runs/tyk2_v2

Incomplete evaluations are skipped. comparison.csv uses the latest budget shared
by both independent policies and all archived controls. final_opdiv_bounds.csv
only contains 10K results. Nothing is extrapolated from an incomplete run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path


LABELS = {
    'synfactor_v2_raw': 'SynFactor + product XGB: raw',
    'synfactor_v2_portfolio': 'SynFactor + product XGB: portfolio',
    'ga_dcso_v14_screening': 'GA-DCSO v14',
    'spacehasten_fast': 'SpaceHASTEN fast',
    'spacehasten_accurate': 'SpaceHASTEN accurate',
    'synfactor_v1': 'SynFactor v1',
}
CORE_METHODS=tuple(LABELS)
BRANCH_METHOD='synfactor_v2_explore_exploit'
LABELS[BRANCH_METHOD]='Explore/exploit + CP fibers (4K inherited)'
EXPECTED = {'synfactor_v2', 'synfactor_v1', 'ga_dcso_v14_screening',
            'spacehasten_fast', 'spacehasten_accurate'}
NUMBERS = ('eligible_molecules','top1_raw_mean','top10_raw_mean','top100_raw_mean',
           'top1000_raw_mean','GPDiv_value','GPDiv_raw_mean','GPDiv_selected_count',
           'OPDiv_value','OPDiv_raw_mean','OPDiv_selected_count',
           'OPDiv_feasible_lower_bound','OPDiv_upper_bound','OPDiv_gap')
FIELDS = ('method','label','target','seed','charged_calls',*NUMBERS,
          'GPDiv_status','OPDiv_status','OPDiv_certified','evidence_count')
CONFIG_FIELDS = ('target','seed','policy','tail_weight','weight_cp','global_k',
                 'local_hits','local_per_hit','random_pool_count','budget',
                 'warm_charged','batch_size','grid_sha256','warm',
                 'druglike_filter','distillation','code_sha256','chiral_xgb',
                 'disagreement_count','fiber_anchor_count','fiber_anchors','fiber_per_slot')


def load(path):
    data=path.read_bytes()
    return json.loads(data),hashlib.sha256(data).hexdigest()


def number(value):
    if value in (None,''):
        return None
    result=float(value)
    if not math.isfinite(result):
        raise ValueError('Aggregate metrics must be finite or empty')
    return result


def equal(a,b,tolerance=1e-8):
    return a is b if a is None or b is None else abs(a-b)<=tolerance


def lineage_identity(cell,config,state,runs):
    """Check the optional fork locally; return only safe aggregate provenance."""
    path=cell/'lineage.json'
    if not path.exists():
        return None
    lineage,lineage_sha=load(path)
    parent_id=lineage.get('parent_run_id','')
    if (path.is_symlink() or not parent_id or Path(parent_id).name!=parent_id
            or lineage.get('schema_version')!=1 or lineage.get('inherited_rounds')!=3
            or lineage.get('inherited_calls')!=4000 or lineage.get('total_budget')!=10000
            or lineage.get('remaining_calls')!=6000 or lineage.get('parent_was_branched') is not False
            or lineage.get('development_fork') is not True or lineage.get('independent_replicate') is not False
            or lineage.get('destination_run_id')!=cell.name or config.get('run_id')!=cell.name
            or config.get('policy')!='raw' or parent_id==cell.name):
        raise ValueError('Unexpected development continuation identity or inherited budget')
    parent=runs/'cells'/parent_id
    parent_config,parent_config_sha=load(parent/'config.json')
    copied_config,copied_config_sha=load(cell/'lineage_parent_config.json')
    if (parent_config.get('policy')!='portfolio' or (parent/'lineage.json').exists()
            or parent_config.get('run_id',parent.name)!=parent_id
            or parent_config_sha!=lineage.get('parent_config_sha256')
            or copied_config_sha!=parent_config_sha or copied_config!=parent_config):
        raise ValueError('Development continuation is not bound to the portfolio reference')
    for key in ('target','seed','budget','warm_charged','batch_size','grid_sha256','warm','oracle_protocol_sha256'):
        if config.get(key)!=parent_config.get(key):
            raise ValueError('Development continuation changed its inherited comparison contract')
    def prefix_digest(value):
        prefix=dict(history=value['history'][:4000],rounds=value['rounds'][:3])
        if len(prefix['history'])!=4000 or len(prefix['rounds'])!=3:
            raise ValueError('Truncated inherited state')
        return hashlib.sha256(json.dumps(prefix,sort_keys=True,allow_nan=False).encode()).hexdigest()
    parent_state,_=load(parent/'state.json')
    if prefix_digest(state)!=lineage.get('prefix_sha256') or prefix_digest(parent_state)!=lineage['prefix_sha256']:
        raise ValueError('Development continuation prefix differs from the actual parent')
    required={f'batch_{i:02d}/{name}' for i in range(1,4)
              for name in ('proposal.json','normalized.json','search.json')}
    if set(lineage.get('batch_sha256',{}))!=required:
        raise ValueError('Incomplete inherited batch provenance')
    for relative in required:
        artifact=cell/relative
        if artifact.is_symlink() or artifact.parent.is_symlink() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=lineage['batch_sha256'][relative]:
            raise ValueError('Inherited evidence changed before aggregate export')
    keys=('parent_run_id','destination_run_id','inherited_rounds','inherited_calls','prefix_sha256',
          'parent_state_sha256','parent_config_sha256','development_fork','independent_replicate')
    return dict(lineage_sha256=lineage_sha,**{key:lineage[key] for key in keys})


def require_lineage_proof(proof,expected):
    if not isinstance(proof,dict) or any(proof.get(key)!=value for key,value in expected.items()):
        raise ValueError('Development continuation evaluation lacks matching verified lineage evidence')


def merge_repeat(previous,new):
    """Repeated control evaluations must agree; intersect valid OPDiv bounds."""
    for key in ('eligible_molecules','top1_raw_mean','top10_raw_mean','top100_raw_mean',
                'top1000_raw_mean','GPDiv_value','GPDiv_raw_mean','GPDiv_selected_count'):
        if not equal(previous[key],new[key]):
            raise ValueError(f'Repeated control differs in {key}; do not average incompatible archives')
    previous['evidence_count']+=1
    lows=[r['OPDiv_feasible_lower_bound'] for r in (previous,new) if r['OPDiv_feasible_lower_bound'] is not None]
    highs=[r['OPDiv_upper_bound'] for r in (previous,new) if r['OPDiv_upper_bound'] is not None]
    if lows:
        lo=max(lows);hi=min(highs) if highs else None
        if hi is not None and lo>hi+1e-6:
            raise ValueError('Repeated full-domain OPDiv intervals conflict')
        if hi is not None:
            hi=max(lo,hi)
        previous.update(OPDiv_value=lo,OPDiv_raw_mean=-lo,OPDiv_feasible_lower_bound=lo,
                        OPDiv_upper_bound=hi,OPDiv_gap=hi-lo if hi is not None else None,
                        OPDiv_selected_count=100,
                        OPDiv_certified=hi is not None and hi-lo<=1e-6,
                        OPDiv_status='optimal' if hi is not None and hi-lo<=1e-6 else 'bounded')


def collect(runs):
    records,provenance,progress,skipped={},[],[],[]
    identity=None
    for cell in sorted((runs/'cells').glob('*')):
        if not (cell/'config.json').exists() or not (cell/'state.json').exists():
            continue
        config,config_sha=load(cell/'config.json');state,state_sha=load(cell/'state.json')
        if config.get('policy') not in ('raw','portfolio'):
            continue
        if (config.get('budget'),config.get('warm_charged'),config.get('batch_size'))!=(10000,1000,1000):
            raise ValueError('Policy does not have its own 10K budget with 1K charged warm start')
        if config.get('druglike_filter') is not False:
            raise ValueError('Only all-valid, unfiltered development runs belong in this report')
        lineage=lineage_identity(cell,config,state,runs)
        policy_method=BRANCH_METHOD if lineage is not None else 'synfactor_v2_'+config['policy']
        progress.append(dict(method=policy_method,policy=config['policy'],charged_calls=len(state['history']),
            lineage=lineage,inherited_calls=4000 if lineage is not None else 0,
            new_calls_after_inherited_prefix=len(state['history'])-4000 if lineage is not None else None,
            completed_adaptive_rounds=len(state.get('rounds',[])),
            local_search_seconds=sum(r['search']['search_seconds'] for r in state.get('rounds',[])),
            timed_remote_oracle_seconds=sum(r['oracle_seconds'] for r in state.get('rounds',[])),
            full_budget_complete=len(state['history'])==10000,
            config={key:config[key] for key in CONFIG_FIELDS if key in config},
            config_sha256=config_sha,state_sha256=state_sha))
        for folder in sorted(cell.glob('evaluation_*k')):
            artifact=f'cells/{cell.name}/{folder.name}'
            if not (folder/'summary.json').exists() or not (folder/'comparison.csv').exists():
                skipped.append(dict(artifact=artifact,reason='evaluation_incomplete'));continue
            summary,summary_sha=load(folder/'summary.json')
            csv_data=(folder/'comparison.csv').read_bytes()
            csv_sha=hashlib.sha256(csv_data).hexdigest()
            rows=list(csv.DictReader(io.StringIO(csv_data.decode())))
            if (len(rows)!=5 or {r['method'] for r in rows}!=EXPECTED
                    or len(summary.get('cells',[]))!=5):
                skipped.append(dict(artifact=artifact,reason='evaluation_incomplete'));continue
            charged=summary['charged_calls']
            if (summary.get('comparison_verified') is not True or charged%1000 or not 1000<=charged<=10000
                    or summary['target']!=config['target'] or summary['seed']!=config['seed']
                    or summary['grid_sha256']!=config['grid_sha256']
                    or summary['warm_sha256']!=config['warm']['csv_sha256']
                    or charged>len(state['history'])):
                raise ValueError(f'Unverified or mismatched evaluation: {artifact}')
            current=tuple(summary[k] for k in ('target','seed','grid_sha256','warm_sha256','rdkit','fingerprint'))
            if identity is not None and current!=identity:
                raise ValueError('Target/seed/grid/warm/RDKit/fingerprint contracts differ')
            identity=current
            by_method={c['method']:c for c in summary['cells']}
            if by_method['synfactor_v2']['identity_evidence'].get('config_sha256')!=config_sha:
                raise ValueError('Evaluated policy configuration differs from its current pinned configuration')
            if lineage is not None:
                if charged<4000:
                    raise ValueError('Continuation cannot claim a separate trajectory before its inherited prefix')
                require_lineage_proof(by_method['synfactor_v2']['identity_evidence'].get('lineage'),lineage)
            for row in rows:
                if int(row['charged_calls'])!=charged:
                    raise ValueError('CSV charged budget differs from verified summary')
                method=policy_method if row['method']=='synfactor_v2' else row['method']
                output=dict(method=method,label=LABELS[method],target=summary['target'],
                            seed=summary['seed'],charged_calls=charged,
                            **{key:number(row.get(key)) for key in NUMBERS},
                            GPDiv_status=row.get('GPDiv_status',''),OPDiv_status=row.get('OPDiv_status',''),
                            OPDiv_certified=str(row.get('OPDiv_certified','')).lower()=='true',evidence_count=1)
                checked=by_method[row['method']]
                if (not equal(output['eligible_molecules'],checked['eligible_molecules'])
                        or not equal(output['top100_raw_mean'],checked['scores']['top100_raw_mean'])):
                    raise ValueError('CSV raw metrics differ from verified summary')
                for portfolio in checked['portfolios']:
                    if not equal(output.get(portfolio['metric']+'_value'),portfolio.get('value')):
                        raise ValueError('CSV portfolio metric differs from verified summary')
                    if portfolio['metric']=='OPDiv':
                        for key in ('feasible_lower_bound','upper_bound','gap','selected_count'):
                            if not equal(output['OPDiv_'+key],portfolio.get(key)):
                                raise ValueError('CSV optimization bound differs from verified summary')
                        if (output['OPDiv_status']!=portfolio.get('status')
                                or output['OPDiv_certified']!=bool(portfolio.get('certified'))):
                            raise ValueError('CSV optimization status differs from verified summary')
                key=(method,charged)
                if key in records:
                    if method.startswith('synfactor_v2_'):
                        raise ValueError('Multiple policy runs would silently replace an independent budget')
                    merge_repeat(records[key],output)
                else:
                    records[key]=output
            provenance.append(dict(artifact=artifact,summary_sha256=summary_sha,
                                   comparison_sha256=csv_sha,config_sha256=config_sha,
                                   method=policy_method,lineage=lineage,
                                   evaluator_source_sha256={Path(k).name:v for k,v in summary.get('source_sha256',{}).items()}))
    ordered=sorted(records.values(),key=lambda r:(r['charged_calls'],list(LABELS).index(r['method'])))
    return ordered,dict(contract=dict(zip(('target','seed','grid_sha256','warm_sha256','rdkit','fingerprint'),identity)) if identity else None,
                        source_evaluations=provenance,policy_progress=progress,skipped=skipped)


def write_csv(path,rows,fields=FIELDS):
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in rows)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(buffer.getvalue());temporary.replace(path)


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    figure,axes=plt.subplots(1,2,figsize=(12.5,5.7))
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#7B61A8','#777777','#222222']
    markers=['o','s','^','D','v','x','P']
    for method,color,marker in zip(LABELS,colors,markers):
        subset=sorted([r for r in rows if r['method']==method],key=lambda r:r['charged_calls'])
        if not subset:
            continue
        for axis,metric in zip(axes,('top100_raw_mean','GPDiv_value')):
            finite=[r for r in subset if r[metric] is not None]
            axis.plot([r['charged_calls']/1000 for r in finite],[r[metric] for r in finite],
                      label=LABELS[method],color=color,marker=marker,markersize=5,
                      linewidth=2.2 if method.startswith('synfactor_v2_') else 1.6,
                      linestyle='-.' if method==BRANCH_METHOD else ('-' if method.startswith('synfactor_v2_') else '--'))
    axes[0].set(title='A  Raw top100: lower is better',ylabel='Mean Glide docking score')
    axes[1].set(title='B  GPDiv100: higher is better',ylabel='Mean utility (negative docking score)')
    budgets=sorted({r['charged_calls']/1000 for r in rows})
    for axis in axes:
        axis.set_xlabel('Charged oracle calls (thousands; includes 1K warm)')
        axis.set_xticks(budgets);axis.grid(alpha=.2);axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    figure.suptitle('TYK2 development, seed 0 — raw, portfolio and development continuation',fontsize=12,y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=9)
    figure.text(.5,.052,'All valid molecules; strict Morgan Tc < 0.30 for GPDiv100. Raw and portfolio references each have a 10K charged-call budget.',
                ha='center',fontsize=8)
    figure.text(.5,.021,'Continuation, when shown: 4K portfolio calls inherited + 6K new calls; adaptive development fork, not an independent replicate.',
                ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.25,1,.94))
    figure.savefig(out/'comparison.png',dpi=220)
    figure.savefig(out/'comparison.svg')
    svg=out/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(figure)


def bound_table(rows,latest,final,branch_latest=None):
    lines=['# OPDiv100 bounds','',
           'Utility is negative mean docking score; larger is better. Every portfolio contains 100 molecules with strict Morgan Tc < 0.30. Feasible bounds are not automatically exact optima.','',
           'Raw and portfolio reference policies are separate runs, each with its own 10,000 charged-call budget. The optional development continuation inherits 4,000 portfolio calls and may add 6,000 new calls; it is not an independent replicate. TYK2 seed 0 is development evidence, not an independent confirmation.','']
    for title,subset in [(f'Latest common evaluated budget: {latest:,} calls' if latest else 'No common evaluated budget',
                          [r for r in rows if r['charged_calls']==latest and r['method'] in CORE_METHODS]),
                         ('Final 10,000-call bounds',final)]:
        lines += ['## '+title,'','| Method | Feasible lower bound | Global upper bound | Status | Certified |',
                  '|---|---:|---:|---|---|']
        by_method={r['method']:r for r in subset}
        methods=CORE_METHODS if title.startswith('Latest') or title.startswith('No common') else LABELS
        for method in methods:
            label=LABELS[method]
            row=by_method.get(method)
            if row is None or row['OPDiv_feasible_lower_bound'] is None:
                lines.append(f'| {label} | — | — | Pending evaluation | No |');continue
            lo=row['OPDiv_feasible_lower_bound'];hi=row['OPDiv_upper_bound']
            lines.append(f"| {label} | {lo:.6f} | {hi:.6f}"+f" | {row['OPDiv_status']} | {'Yes' if row['OPDiv_certified'] else 'No'} |" if hi is not None
                         else f"| {label} | {lo:.6f} | — | {row['OPDiv_status']} | No |")
        lines.append('')
    if branch_latest is not None:
        row=next(r for r in rows if r['method']==BRANCH_METHOD and r['charged_calls']==branch_latest)
        lines += ['## Development continuation','',
                  f'Latest matched prefix: {branch_latest:,} charged calls, including the inherited 4,000. '
                  'See development_branch_comparison.csv for all seven series at each available matched branch budget.',
                  f"Continuation OPDiv100 interval: [{row['OPDiv_feasible_lower_bound']}, {row['OPDiv_upper_bound']}]; "
                  f"status: {row['OPDiv_status']}. This branch must not be counted as a third independent replicate.",'']
    lines += ['Repeated baseline evaluations must agree on raw/GPDiv metrics. Their valid full-domain OPDiv intervals are intersected (largest feasible lower bound, smallest upper bound); no restricted-domain bound is used.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/tyk2_v2'))
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    rows,evidence=collect(args.runs)
    if not rows:
        raise SystemExit('No complete verified evaluations available')
    common=set.intersection(*[{r['charged_calls'] for r in rows if r['method']==method} for method in CORE_METHODS])
    latest=max(common) if common else None
    branch_common=common & {r['charged_calls'] for r in rows if r['method']==BRANCH_METHOD}
    branch_latest=max(branch_common) if branch_common else None
    final=[r for r in rows if r['charged_calls']==10000]
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'trajectories.csv',rows)
    write_csv(args.out/'comparison.csv',[r for r in rows if r['charged_calls']==latest and r['method'] in CORE_METHODS])
    write_csv(args.out/'development_branch_comparison.csv',[r for r in rows if r['charged_calls'] in branch_common])
    write_csv(args.out/'final_opdiv_bounds.csv',final,
              ('method','label','target','seed','charged_calls','OPDiv_feasible_lower_bound','OPDiv_upper_bound',
               'OPDiv_gap','OPDiv_selected_count','OPDiv_status','OPDiv_certified'))
    timings=[dict(method=p['method'],label=LABELS[p['method']],charged_calls=p['charged_calls'],
                  search_seconds=p['local_search_seconds'],oracle_seconds=p['timed_remote_oracle_seconds'],
                  timed_stage_sum_seconds=p['local_search_seconds']+p['timed_remote_oracle_seconds'])
             for p in evidence['policy_progress']]
    write_csv(args.out/'timings.csv',timings,
              ('method','label','charged_calls','search_seconds','oracle_seconds','timed_stage_sum_seconds'))
    (args.out/'opdiv_bounds.md').write_text(bound_table(rows,latest,final,branch_latest))
    evidence.update(updated_utc=datetime.now(timezone.utc).isoformat(),latest_common_evaluated_budget=latest,
                    latest_matched_development_branch_budget=branch_latest,
                    reference_policy_budgets_are_separate=True,development_branch_is_independent=False,
                    development_evidence_only=True,
                    raw_molecules_exported=False,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    timing_scope='Local search and timed remote oracle only; excludes initialization/preflight/transport and untimed remote assembly')
    (args.out/'provenance.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    plots(rows,args.out)
    print(json.dumps(dict(trajectory_rows=len(rows),latest_common_evaluated_budget=latest,
                          final_10k_methods=len(final),skipped_incomplete=len(evidence['skipped']),output=str(args.out)),indent=2))


if __name__=='__main__':
    main()
