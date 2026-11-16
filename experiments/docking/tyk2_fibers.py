"""Optional CP-optimized one-slot analogues of currently observed molecules.

Anchors are selected solely from the supplied charged history. A cached CP
matrix-vector product scores each full compatible fiber; only top candidates
are returned for independent product-XGB reranking. No reaction quotas,
archival winner seeds, assembly or docking are used.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def observed_anchors(history, *, policy='raw', count=100, fingerprints=None, threshold=.30):
    if policy not in ('raw','gpdiv') or count<1 or not 0<threshold<=1:
        raise ValueError('Need raw/gpdiv policy, positive anchor count and a valid strict similarity threshold')
    if fingerprints is not None and len(fingerprints)!=len(history):
        raise ValueError('Observed fingerprints must align with the complete charged history')
    ids=[r['product_id'] for r in history]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate charged route identities')
    valid=[]
    for i,row in enumerate(history):
        try: score=float(row.get('score'))
        except (TypeError,ValueError): continue
        if row.get('status')=='valid' and np.isfinite(score) and abs(score)<100:
            valid.append(i)
    valid.sort(key=lambda i:(float(history[i]['score']),ids[i]))
    selected,chosen_fps=[],[]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    for i in valid:
        if policy=='gpdiv':
            fp=fingerprints[i] if fingerprints is not None else None
            if fingerprints is None:
                smiles=history[i].get('smiles')
                mol=Chem.MolFromSmiles(smiles) if isinstance(smiles,str) else None
                fp=generator.GetFingerprint(mol) if mol is not None else None
            if fp is None:
                continue
            if chosen_fps and max(DataStructs.BulkTanimotoSimilarity(fp,chosen_fps))>=threshold:
                continue
            chosen_fps.append(fp)
        selected.append(i)
        if len(selected)==count:
            break
    return np.asarray(selected,np.int64)


def _key(row,rid):
    return (int(rid),*map(int,row))


@torch.no_grad()
def cp_fiber_candidates(space,model,history, *, anchor_policy='raw',anchor_count=100,
                        per_fiber=256,max_candidates=100000,observed_fingerprints=None,
                        anchor_threshold=.30,excluded_ids=(),block_size=65536,max_seconds=30):
    """Exact topK within each observed-anchor/variable-slot fiber.

    All remaining slots stay fixed to the anchor, and every compatible synthon
    is scored. Memory scales with block_size*rank plus the retained shortlist.
    Ties choose the smaller global synthon row. Float32 MVM can differ from
    elementwise model.predict accumulation by normal floating-point roundoff.
    Overlapping fibers are deduplicated by route identity, retaining first
    anchor provenance. max_candidates is a strict capacity check, not a score
    truncation that could silently erase later anchors.
    """
    if min(anchor_count,per_fiber,max_candidates,block_size)<1 or (max_seconds is not None and max_seconds<=0):
        raise ValueError('Counts and time limit must be positive')
    if model.model is None or not hasattr(model,'cached_factors'):
        raise RuntimeError('Fit the CP model and cache its factors first')
    started=time.perf_counter()
    anchors=observed_anchors(history,policy=anchor_policy,count=anchor_count,
        fingerprints=observed_fingerprints,threshold=anchor_threshold)
    if not len(anchors):
        raise ValueError('No valid observed anchors')
    anchor_ids=[history[i]['product_id'] for i in anchors]
    rows,rids=space.decode(anchor_ids)
    grids={int(g['reaction_index']):tuple(np.sort(np.asarray(a,np.int64)) for a in g['slots']) for g in space.grids}
    fibers=sum(len(grids[int(r)]) for r in rids)
    if fibers*per_fiber>max_candidates:
        raise ValueError('Requested per-fiber capacity exceeds max_candidates; lower per_fiber or anchor_count explicitly')
    observed_ids=[r['product_id'] for r in history]
    excluded_rows,excluded_rids=space.decode(list(dict.fromkeys(observed_ids+list(excluded_ids))))
    seen={_key(row,rid) for row,rid in zip(excluded_rows,excluded_rids)}
    # For a fiber, all non-varying components identify the matching exclusions.
    exclusions={}
    for row,rid in zip(excluded_rows,excluded_rids):
        for slot in range(len(grids[int(rid)])):
            key=(int(rid),slot,*[int(v) for j,v in enumerate(row) if j!=slot])
            exclusions.setdefault(key,set()).add(int(row[slot]))
    factors,additive=model.cached_factors,model.cached_additive
    device=factors.device
    if factors.dtype!=torch.float32 or not np.isfinite(model.y_scale) or model.y_scale<=0:
        raise ValueError('Need float32 factors and a positive finite target scale')
    output_rows,output_rids,output_scores,parent_ids,variable_slots=[],[],[],[],[]
    visited=0;blocks=0;per_anchor=[]
    previous_precision=torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision='ieee'
    try:
        for anchor,row,rid in zip(anchor_ids,rows,rids):
            rid=int(rid);axes=grids[rid];before=len(output_rows)
            for slot,axis in enumerate(axes):
                fixed=torch.ones(model.rank,dtype=torch.float32,device=device)
                bias=model.model.reaction_bias.weight[rid,0]+model.model.bias
                for j in range(len(axes)):
                    if j!=slot:
                        fixed=fixed*factors[j,int(row[j])]
                        if additive is not None:
                            bias=bias+additive[j,int(row[j])]
                excluded=exclusions.get((rid,slot,*[int(v) for j,v in enumerate(row) if j!=slot]),set())
                best_values=np.empty(0,np.float32);best_rows=np.empty(0,np.int64)
                for start in range(0,len(axis),block_size):
                    part=axis[start:start+block_size]
                    indices=torch.as_tensor(part,device=device)
                    values=(factors[slot,indices]@fixed)/math.sqrt(model.rank)+bias
                    if additive is not None:values=values+additive[slot,indices]
                    values=values*model.y_scale+model.y_mean
                    if not torch.isfinite(values).all():
                        raise FloatingPointError('Nonfinite CP fiber predictions')
                    # Only a bounded fiber block is copied; CPU stable sorting
                    # gives exact deterministic membership at tied boundaries.
                    scores=values.cpu().numpy()
                    allowed=~np.isin(part,np.fromiter(excluded,np.int64)) if excluded else np.ones(len(part),bool)
                    candidate_rows=np.concatenate([best_rows,part[allowed]])
                    candidate_scores=np.concatenate([best_values,scores[allowed]])
                    order=np.lexsort((candidate_rows,-candidate_scores))[:per_fiber]
                    best_rows,best_values=candidate_rows[order],candidate_scores[order]
                    visited+=len(part);blocks+=1
                    if max_seconds is not None and time.perf_counter()-started>max_seconds:
                        raise TimeoutError(f'Incomplete CP fiber scan: {visited} synthon substitutions scored; no partial shortlist returned')
                for replacement,value in zip(best_rows,best_values):
                    candidate=row.copy();candidate[slot]=replacement
                    key=_key(candidate,rid)
                    if key in seen:continue
                    seen.add(key);output_rows.append(candidate);output_rids.append(rid);output_scores.append(float(value))
                    parent_ids.append(anchor);variable_slots.append(slot)
            per_anchor.append(dict(product_id=anchor,reaction_index=rid,added=len(output_rows)-before))
    finally:
        torch.backends.cuda.matmul.fp32_precision=previous_precision
    tuples=np.asarray(output_rows,np.int64).reshape(-1,3);reactions=np.asarray(output_rids,np.int64)
    scores=np.asarray(output_scores,np.float32)
    return dict(product_ids=space.ids(tuples,reactions),tuples=tuples,reaction_ids=reactions,cp_scores=scores,
        source=['cp_observed_fiber']*len(tuples),parent_product_ids=parent_ids,variable_slots=np.asarray(variable_slots,np.int64),
        diagnostics=dict(anchor_policy=anchor_policy,anchor_requested=anchor_count,anchor_count=len(anchors),
            anchor_history_indices=anchors.tolist(),anchor_product_ids=anchor_ids,
            anchor_similarity_threshold=anchor_threshold if anchor_policy=='gpdiv' else None,
            anchor_reactions=len(set(map(int,rids))),returned_reactions=len(set(output_rids)),
            fibers=fibers,per_fiber=per_fiber,visited_substitutions=visited,blocks=blocks,
            returned_routes=len(tuples),per_anchor=per_anchor,complete=True,
            seconds=time.perf_counter()-started,ranking='CP utility descending; global synthon row ascending',
            precision='float32 MVM IEEE; includes additive, reaction offset and target normalization'))
