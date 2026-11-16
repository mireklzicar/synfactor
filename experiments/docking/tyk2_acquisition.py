"""Observed-label-only candidate generation and independent-model acquisition.

CP proposes routes; an independently fitted full-product XGB model ranks them.
No docking, model fitting, teacher distillation or archival winner lookup occurs
here. Utilities are higher-is-better. Reaction quotas are never imposed.
"""
from __future__ import annotations

import math
import time

import numpy as np
from rdkit import DataStructs

from experiments.docking.cartesian import stream_topk


def _key(row, rid):
    return (int(rid), *map(int, row))


def _nearest_synthons(features, axis, anchor, count, block_size=4096):
    """Binary Morgan512 Tanimoto for local proposals, never product diversity."""
    anchor_bits = np.asarray(features[anchor, :512]) > 0
    anchor_count = int(anchor_bits.sum())
    similarities = np.empty(len(axis), np.float32)
    for start in range(0, len(axis), block_size):
        bits = np.asarray(features[axis[start:start+block_size], :512]) > 0
        intersection = bits[:, anchor_bits].sum(1)
        union = bits.sum(1)+anchor_count-intersection
        similarities[start:start+len(bits)] = np.divide(intersection, union,
            out=np.zeros(len(bits), dtype=np.float64), where=union>0)
    order = np.lexsort((axis, -similarities))
    return axis[order[axis[order] != anchor][:count]]


def observed_neighbors(space, history, *, hits=100, per_hit=128, seed=0, excluded=()):
    """One-slot substitutions around only valid, already observed best hits.

    Half the proposals use nearest capped-synthon Morgan512 neighbors; half use
    random compatible substitutions. All proposals remain in the hit's reaction.
    This is an observed-hit local search policy, not a per-reaction quota.
    """
    if hits < 0 or per_hit < 0:
        raise ValueError('Neighbor counts must be nonnegative')
    ids = [r['product_id'] for r in history]
    if len(set(ids)) != len(ids):
        raise ValueError('Observed history contains duplicate route identities')
    observed, reactions = space.decode(ids)
    seen = set(excluded) | {_key(row,rid) for row,rid in zip(observed,reactions)}
    valid = [i for i,r in enumerate(history) if r.get('status') == 'valid'
             and r.get('score') is not None and np.isfinite(float(r['score']))
             and abs(float(r['score']))<100]
    valid.sort(key=lambda i:(float(history[i]['score']),ids[i]))
    rng = np.random.default_rng(seed)
    rows, rids, parents = [], [], []
    cache = {}
    grids = {int(g['reaction_index']):g['slots'] for g in space.grids}
    for i in valid[:hits]:
        rid, parent = int(reactions[i]), observed[i]
        axes = grids[rid]
        nearest_per_slot = math.ceil(per_hit/2/len(axes))
        candidates = []
        for slot,axis in enumerate(axes):
            axis = np.asarray(axis,np.int64)
            key = (rid,slot,int(parent[slot]))
            if key not in cache:
                cache[key] = _nearest_synthons(space.features,axis,int(parent[slot]),nearest_per_slot)
            candidates.extend((slot,int(value)) for value in cache[key])
        # Interleave slots so truncation does not systematically favor slot1.
        rng.shuffle(candidates)
        candidates = candidates[:math.ceil(per_hit/2)]
        for _ in range(per_hit*3):
            slot = int(rng.integers(len(axes)))
            candidates.append((slot,int(rng.choice(axes[slot]))))
        count = 0
        for slot,value in candidates:
            if value == parent[slot]:
                continue
            row = parent.copy();row[slot] = value
            key = _key(row,rid)
            if key in seen:
                continue
            rows.append(row);rids.append(rid);parents.append(ids[i]);seen.add(key)
            count += 1
            if count == per_hit:
                break
    return dict(tuples=np.asarray(rows,np.int64).reshape(-1,3),
                reaction_ids=np.asarray(rids,np.int64), parent_product_ids=parents)


def build_candidate_pool(space, cp_models, history, *, global_k=100000, local_hits=100,
                         local_per_hit=128, random_pool_count=5000, seed=0,
                         max_sweep_seconds=300, block_rows=4096, block_cols=8192):
    """Merge global CP top routes, observed-hit neighbors and uniform routes.

    A sequence of CP models divides the global shortlist capacity equally across
    models. This optional union does not reserve any slots for reaction families.
    A sweep timeout propagates: an incomplete sweep is never called global topK.
    """
    started = time.perf_counter()
    models = list(cp_models) if isinstance(cp_models,(list,tuple)) else [cp_models]
    if not models or global_k < 0 or random_pool_count < 0:
        raise ValueError('Need models and nonnegative pool counts')
    observed_ids = [r['product_id'] for r in history]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError('Observed history contains duplicate route identities')
    observed, observed_rids = space.decode(observed_ids)
    seen = {_key(row,rid) for row,rid in zip(observed,observed_rids)}
    total = int(space.manifest['products'])
    rows, rids, sources, parents, sweeps = [], [], [], [], []
    def append(tuples,reactions,source,parent_ids=None):
        for i,(row,rid) in enumerate(zip(tuples,reactions)):
            key = _key(row,rid)
            if key in seen:
                continue
            seen.add(key);rows.append(row);rids.append(int(rid));sources.append(source)
            parents.append(parent_ids[i] if parent_ids is not None else None)
    for index,model in enumerate(models):
        k = global_k//len(models)+(index < global_k%len(models))
        k = min(k,total-len(observed))
        if k:
            sweep = stream_topk(model,space.grids,k=k,excluded_tuples=observed,
                excluded_reaction_ids=observed_rids,block_rows=block_rows,block_cols=block_cols,
                max_seconds=max_sweep_seconds)
            append(sweep['tuples'],sweep['reaction_ids'],f'global_cp_{index}')
            sweeps.append(sweep['diagnostics'])
    local = observed_neighbors(space,history,hits=local_hits,per_hit=local_per_hit,
                               seed=seed+17,excluded=seen)
    append(local['tuples'],local['reaction_ids'],'observed_hit_neighbor',local['parent_product_ids'])
    random_count = min(random_pool_count,max(0,total-len(seen)))
    if random_count:
        random_rows,random_rids = space.sample(random_count,np.random.default_rng(seed+101),seen)
        append(random_rows,random_rids,'uniform_route')
    tuples = np.asarray(rows,np.int64).reshape(-1,3)
    reactions = np.asarray(rids,np.int64)
    if not len(rows):
        raise ValueError('Candidate pool is empty after excluding observed routes')
    predictions = np.column_stack([m.predict(tuples,reactions) for m in models])
    if predictions.shape != (len(rows),len(models)) or not np.isfinite(predictions).all():
        raise ValueError('CP predictions must be finite and aligned with candidate routes')
    return dict(product_ids=space.ids(tuples,reactions),tuples=tuples,reaction_ids=reactions,
        cp_scores=predictions.mean(1),cp_predictions=predictions,source=sources,parent_product_ids=parents,
        diagnostics=dict(pool_size=len(rows),observed_count=len(history),sweeps=sweeps,
            source_counts={s:sources.count(s) for s in sorted(set(sources))},
            reaction_count=len(np.unique(reactions)),seconds=time.perf_counter()-started))


def _percentiles(values):
    """Average-tie percentile ranks; raw outlier magnitude cannot dominate."""
    values = np.asarray(values,float)
    order = np.argsort(values,kind='stable')
    sorted_values = values[order]
    starts = np.r_[0,np.flatnonzero(np.diff(sorted_values))+1]
    ends = np.r_[starts[1:],len(values)]
    result = np.empty(len(values),float)
    for start,end in zip(starts,ends):
        result[order[start:end]] = ((start+end-1)/2)/max(1,len(values)-1)
    return result


def select_batch(product_ids, xgb_scores, *, n_exploit=800, n_disagreement=100, n_random=100,
                 cp_scores=None, product_fingerprints=None, molecule_keys=None,
                 excluded_ids=(), excluded_molecule_keys=(), reference_fingerprints=(),
                 mode='raw', max_similarity=.30, novelty_weight=.35,
                 disagreement_quality_quantile=.50, seed=0, gpdiv_fallback='raise'):
    """Select aligned pool indices using independent full-product XGB utilities.

    raw: score ranking with no analogue suppression. novelty: greedy percentile
    quality minus novelty_weight * maximum true-product Tc. gpdiv: score-ordered
    greedy selection requiring every Tc strictly below max_similarity; shortage
    raises unless gpdiv_fallback='ranked' explicitly permits and records filling.

    Exploration is bounded CP/XGB rank disagreement in the XGB-qualified pool,
    followed by uniform random pool candidates. It is not a calibrated estimate
    of mutual information. Optional molecule_keys enforce canonical uniqueness.
    """
    ids = list(product_ids);n = len(ids);scores = np.asarray(xgb_scores,float)
    counts = (n_exploit,n_disagreement,n_random)
    if (scores.shape != (n,) or len(set(ids)) != n or any(c<0 or int(c)!=c for c in counts)
            or mode not in ('raw','novelty','gpdiv') or gpdiv_fallback not in ('raise','ranked')
            or not 0<=max_similarity<=1 or not 0<=disagreement_quality_quantile<=1 or novelty_weight<0):
        raise ValueError('Invalid acquisition inputs/configuration')
    if mode != 'raw' and product_fingerprints is None:
        raise ValueError('Diversity requires true full-product fingerprints')
    fingerprints = list(product_fingerprints) if product_fingerprints is not None else None
    keys = list(molecule_keys) if molecule_keys is not None else ids
    if len(keys)!=n or (fingerprints is not None and len(fingerprints)!=n):
        raise ValueError('Molecular identities/fingerprints must align with candidate IDs')
    excluded = set(excluded_ids);seen_keys = set(excluded_molecule_keys)
    eligible = np.isfinite(scores) & np.array([pid not in excluded for pid in ids])
    if fingerprints is not None:
        eligible &= np.array([fp is not None for fp in fingerprints])
    eligible &= np.array([key is not None and key not in seen_keys for key in keys])
    rank = np.zeros(n);rank[eligible] = _percentiles(scores[eligible]) if eligible.any() else 0
    chosen, labels = [], []
    similarities = np.zeros(n)
    usable = np.flatnonzero(eligible)
    usable_fps = [fingerprints[i] for i in usable] if fingerprints is not None else None
    key_groups = {}
    for i in usable:
        key_groups.setdefault(keys[i],[]).append(int(i))
    def update_similarity(fp):
        similarities[usable] = np.maximum(similarities[usable],
            DataStructs.BulkTanimotoSimilarity(fp,usable_fps))
    if mode != 'raw':
        for fp in reference_fingerprints:
            if fp is not None:
                update_similarity(fp)
    def admit(index,label):
        if not eligible[index] or keys[index] in seen_keys:
            return False
        chosen.append(int(index));labels.append(label);seen_keys.add(keys[index]);eligible[key_groups[keys[index]]]=False
        if mode != 'raw':
            update_similarity(fingerprints[index])
        return True
    def take_ranked(order,count,label):
        added = 0
        for index in order:
            if admit(int(index),label):
                added += 1
                if added == count:
                    break
        return added
    fallback = 0
    if n_exploit:
        if mode == 'novelty':
            for _ in range(n_exploit):
                available = np.flatnonzero(eligible)
                if not len(available):
                    break
                quality = rank[available]-novelty_weight*similarities[available]
                ties = available[quality==quality.max()]
                best = ties[np.argmax(scores[ties])]
                admit(int(best),'exploit')
        else:
            order = np.lexsort((np.arange(n),-np.nan_to_num(scores,nan=-np.inf)))
            for index in order:
                if mode == 'gpdiv' and similarities[index]>=max_similarity:
                    continue
                admit(int(index),'exploit')
                if len(chosen)==n_exploit:
                    break
            if len(chosen)<n_exploit and mode=='gpdiv' and gpdiv_fallback=='ranked':
                fallback = take_ranked(order,n_exploit-len(chosen),'exploit_ranked_fallback')
    if len(chosen)!=n_exploit:
        raise ValueError(f'Only {len(chosen)}/{n_exploit} exploitation candidates satisfy the selected policy')
    if n_disagreement:
        cp = np.asarray(cp_scores,float)
        if cp.shape != (n,) or not np.isfinite(cp[usable]).all():
            raise ValueError('Disagreement requires aligned finite independent CP predictions')
        # Rank across original usable candidates, not the changing selected set.
        cp_rank = np.zeros(n);cp_rank[usable] = _percentiles(cp[usable])
        disagreement = np.abs(cp_rank-rank)
        qualified = eligible & (rank>=disagreement_quality_quantile)
        positions = np.flatnonzero(qualified)
        order = positions[np.lexsort((positions,-scores[positions],-(disagreement[positions]+.1*rank[positions])))]
        added = take_ranked(order,n_disagreement,'rank_disagreement')
        if added != n_disagreement:
            raise ValueError(f'Only {added}/{n_disagreement} distinct quality-qualified disagreement candidates')
    if n_random:
        added = take_ranked(np.random.default_rng(seed).permutation(np.flatnonzero(eligible)),n_random,'random')
        if added != n_random:
            raise ValueError(f'Only {added}/{n_random} random candidates remain')
    return dict(indices=np.asarray(chosen,np.int64),product_ids=[ids[i] for i in chosen],source=labels,
        diagnostics=dict(mode=mode,counts={label:labels.count(label) for label in sorted(set(labels))},
            max_similarity=max_similarity if mode=='gpdiv' else None,novelty_weight=novelty_weight if mode=='novelty' else None,
            gpdiv_ranked_fallback=fallback,diversity_applies_to='exploitation subset and supplied references',
            disagreement='absolute CP versus independent XGB percentile difference plus0.1 XGB percentile',
            disagreement_quality_quantile=disagreement_quality_quantile))
