"""Aggregate acquisition-arm diagnostics from an existing live cell; no fitting.

Reads private local state/evaluation artifacts but emits no structures or route
IDs. Arm comparisons are descriptive and selection-biased. Portfolio ranks use
the evaluator's canonical-molecule representatives; failures remain charged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ('warm', 'exploit', 'information', 'random')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return float(values.mean()) if len(values) else None


def minimum(values):
    return float(values.min()) if len(values) else None


def analyze(cell):
    state_path, config_path = cell/'state.json', cell/'config.json'
    state, config = json.loads(state_path.read_text()), json.loads(config_path.read_text())
    history = pd.DataFrame(state['history'])
    history['score'] = pd.to_numeric(history.score, errors='coerce')
    history['reaction'] = history.product_id.str.split('____').str[0]
    history['arm'], history['prediction'], history['conditional_gain'] = 'warm', np.nan, np.nan
    warm_count = int(config['warm_charged'])
    assert len(history) == int(config['budget']) == history.product_id.nunique()
    assert (history.iloc[:warm_count].iteration < 0).all()
    seen = set(history.iloc[:warm_count].reaction)
    warm_reactions = set(seen)
    discoveries, cold_gains, cold_expected, round_summaries = {a: 0 for a in ARMS}, [], [], []
    offset = warm_count
    for record in state['rounds']:
        search = record['search']; iteration = record['iteration']
        counts = {arm: int(search[arm+'_count']) for arm in ARMS[1:]}
        size = sum(counts.values())
        assert len(search['predictions']) == size
        assert (history.iloc[offset:offset+size].iteration == iteration).all()
        history.loc[offset:offset+size-1, 'prediction'] = search['predictions']
        info = search['information']; assert len(info['conditional_gain_nats']) == counts['information']
        before = set(seen); start = offset
        per_arm = {}
        for arm, count in counts.items():
            history.loc[start:start+count-1, 'arm'] = arm
            if arm == 'information':
                history.loc[start:start+count-1, 'conditional_gain'] = info['conditional_gain_nats']
            for local, reaction in enumerate(history.iloc[start:start+count].reaction):
                if reaction not in seen:
                    discoveries[arm] += 1; seen.add(reaction)
                    if arm == 'information':
                        cold_gains.append(info['conditional_gain_nats'][local])
                        cold_expected.append(.5*math.log1p(1e6/info['noise_variance']))
            block = history.iloc[start:start+count]
            valid = block[(block.status == 'valid') & np.isfinite(block.score)]
            per_arm[arm] = dict(calls=count, valid=len(valid), valid_fraction=len(valid)/count,
                mean_valid_score=mean(valid.score), best_valid_score=minimum(valid.score),
                reactions=int(block.reaction.nunique()),
                mean_valid_prediction=mean(valid.prediction),
                mean_valid_prediction_optimism=mean(valid.prediction+valid.score))
            start += count
        fit = search['fit']
        round_summaries.append(dict(iteration=iteration, acquired_before=offset,
            validation_nmse=fit['best_val_nmse'], selected_epoch=fit['selected_epoch'],
            training_epochs=fit['epochs'], validation_count=fit['validation_count'],
            information_selected=counts['information'], information_gain_nats=info['total_conditional_gain_nats'],
            information_noise_variance=info['noise_variance'], reactions_before=len(before),
            newly_queried_reactions=len(seen-before), reactions_after=len(seen), arms=per_arm))
        offset += size
    assert offset == len(history)
    paths = [state_path, config_path, cell/'summary.json', cell/'evaluation_live/summary.json']
    domains = {}
    for mode in ('all-valid', 'druglike-v1'):
        path = cell/f'evaluation_live/synfactor__{mode}__molecules.csv.gz'; paths.append(path)
        frame = pd.read_csv(path).merge(history[['product_id','arm','reaction']], on='product_id', validate='1:1')
        # Do not retain the source SMILES/IDs in the aggregate output.
        frame = frame.sort_values(['score','product_id'], kind='stable')
        domain = dict(unique_molecules=len(frame), arms={}, top_contributions={})
        for arm in ARMS:
            subset = frame[frame.arm == arm]
            domain['arms'][arm] = dict(unique_molecules=len(subset),
                fraction_of_charged_arm=len(subset)/int((history.arm == arm).sum()),
                mean_score=mean(subset.score), best_score=minimum(subset.score))
        for k in (100, 1000):
            top = frame.iloc[:k]
            domain['top_contributions'][str(k)] = dict(requested=k, available=len(top),
                complete=len(top)==k, by_arm={a:int((top.arm == a).sum()) for a in ARMS},
                molecules_from_reactions_absent_in_warm=int((~top.reaction.isin(warm_reactions)).sum()))
        domains[mode] = domain
    arms = {}
    for arm in ARMS:
        subset = history[history.arm == arm]
        valid = subset[(subset.status == 'valid') & np.isfinite(subset.score)]
        frequencies = subset.reaction.value_counts()
        novel = valid[~valid.reaction.isin(warm_reactions)]
        arms[arm] = dict(calls=len(subset), valid=len(valid), valid_fraction=len(valid)/len(subset),
            mean_valid_score=mean(valid.score), median_valid_score=float(valid.score.median()),
            best_valid_score=minimum(valid.score), reactions=int(subset.reaction.nunique()),
            dominant_reaction_fraction=float(frequencies.iloc[0]/len(subset)),
            valid_queries_in_reactions_absent_in_warm=len(novel),
            best_score_in_reactions_absent_in_warm=minimum(novel.score))
        if arm != 'warm':
            arms[arm]['mean_valid_prediction_optimism'] = mean(valid.prediction+valid.score)
    valid_all = pd.read_csv(cell/'evaluation_live/synfactor__all-valid__molecules.csv.gz').merge(
        history[['product_id','arm']], on='product_id', validate='1:1')
    failures = {a:valid_all.loc[valid_all.arm == a,'druglike_failure'].dropna().value_counts().astype(int).to_dict() for a in ARMS}
    total_gain = sum(r['information_gain_nats'] for r in round_summaries)
    dimension = state['rounds'][0]['search']['information']['dimension']
    return dict(target=config['target'], seed=config['seed'], charged_calls=len(history),
        source_sha256={str(p):digest(p) for p in paths}, script_sha256=digest(Path(__file__)),
        source_code_sha256=config.get('code_sha256', {}), arm_inference='Ordered batch positions from recorded exploit/information/random counts; normalized oracle order preserved',
        caveats=['Arm means are valid-only and selection-biased, not causal comparisons.',
                 'Information gains sum across changing fitted representations; not a single neural posterior.',
                 'Validation partitions and acquired-data distributions change between rounds.',
                 'Domain contributions use best-score canonical-molecule representatives from existing evaluation.'],
        arms=arms, domains=domains, druglike_first_failure_counts=failures,
        reaction_coverage=dict(warm=len(warm_reactions), final=len(seen),
            available=dimension-int(config['rank'])-(3 if config['additive'] else 0)-1,
            first_new_reaction_discoveries_by_arm=discoveries,
            reactions_queried_only_by_information=len(set(history[history.arm=='information'].reaction)-set(history[history.arm!='information'].reaction))),
        information=dict(selected=sum(r['information_selected'] for r in round_summaries),
            total_gain_nats=total_gain, cold_reaction_selections=len(cold_gains),
            cold_reaction_gain_nats=sum(cold_gains), cold_reaction_fraction_of_gain=sum(cold_gains)/total_gain,
            cold_gain_scale_explanation='An unseen one-hot column has RMS floor 0.001, implying prior feature variance 1e6 with precision 1. Its gain is approximately 0.5*log(1+1e6/noise_variance).',
            cold_gain_formula_max_absolute_error=float(np.max(np.abs(np.array(cold_gains)-cold_expected))) if cold_gains else 0.),
        rounds=round_summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('arm_diagnostics.json'))
    args = parser.parse_args(); result = analyze(args.cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved aggregate diagnostics to {args.out}; {result["charged_calls"]} calls; no structures exported')


if __name__ == '__main__':
    main()
