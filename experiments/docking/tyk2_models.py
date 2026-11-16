"""Independent direct-docking CP and true-product XGB models for TYK2 v2.

Neither regressor takes predictions from the other as training targets. Label
selection follows the archived GA-v14 Glide policy: valid finite |score|<100,
with failures still charged by the caller. XGB uses the exact v14 XGB preset;
full-product Morgan replaces its synthon-OR representation. Defaults retain
uniform weights and the original unconditioned CP architecture.
"""
from __future__ import annotations

import copy
import math
import time
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from torch import nn

from .factor_model import FactorRegressor, _FactorNetwork

XGB_VERSION = '3.2.0'  # Installed production Python3.11 runtime; explicit CPU wheel.
GA_V14_XGB_PARAMS = dict(n_estimators=256, max_depth=8, learning_rate=.05,
    subsample=.8, colsample_bytree=.8, objective='reg:squarederror',
    tree_method='hist', verbosity=0)


def training_labels(history):
    """Return best-first acquired row indices and observed higher-better utilities."""
    selected = []
    for i, row in enumerate(history):
        try:
            score = float(row.get('score'))
        except (TypeError, ValueError):
            continue
        if row.get('status') == 'valid' and math.isfinite(score) and abs(score) < 100:
            selected.append((i, -score))
    selected.sort(key=lambda pair: -pair[1])  # same stable order as GA-v14 filter
    return np.array([i for i, _ in selected], dtype=np.int64), np.array([y for _, y in selected], dtype=np.float32)


def tail_weights(y, maximum=2., power=4.):
    """Gentle rank-tail emphasis, mean1, max/min<=maximum; every row retained.

    Evaluate only on acquired utilities. Uniform default is maximum1. A fixed
    maximum2 gives a gentle variant; maximum4 is the strongest allowed variant.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Need finite acquired utilities')
    if not 1 <= maximum <= 4 or not math.isfinite(power) or power <= 0:
        raise ValueError('Need maximum in[1,4] and positive finite power')
    percentile = (rankdata(y, method='average')-.5)/len(y)
    weights = 1.+(maximum-1.)*percentile**power
    return (weights/weights.mean()).astype(np.float32)


def product_morgan(smiles, *, n_jobs=4, batch_size=4096):
    """True product Morgan radius2/2048/nonchiral bits; no standardization."""
    smiles = list(smiles)
    if n_jobs < 1 or batch_size < 1:
        raise ValueError('Need positive thread and chunk counts')
    result = np.empty((len(smiles), 2048), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    for start in range(0, len(smiles), batch_size):
        molecules = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles[start:start+batch_size]]
        if any(mol is None for mol in molecules):
            raise ValueError(f'Invalid product SMILES in chunk starting at{start}; never silently use zero bits')
        fps = generator.GetFingerprints(molecules, numThreads=n_jobs)
        for j, fp in enumerate(fps, start):
            DataStructs.ConvertToNumpyArray(fp, result[j])
    return result


class ProductXGBRegressor:
    """GA-v14 tree hyperparameters with independently observed docking labels."""
    def __init__(self, seed=0, n_jobs=4):
        self.seed, self.n_jobs, self.model = int(seed), int(n_jobs), None
        if self.n_jobs < 1:
            raise ValueError('n_jobs must be positive')

    @staticmethod
    def _features(features):
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != 2048 or not np.isfinite(x).all():
            raise ValueError('Need finite [n,2048] Morgan bit features')
        if np.any((x != 0) & (x != 1)):
            raise ValueError('Expected binary Morgan bits')
        return x

    def fit_features(self, features, y, sample_weight=None):
        import xgboost
        if xgboost.__version__ != XGB_VERSION:
            raise RuntimeError(f'Use xgboost-cpu=={XGB_VERSION}; found{xgboost.__version__}')
        x = self._features(features); y = np.asarray(y, np.float32)
        if y.shape != (len(x),) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError('Need at least two finite acquired labels')
        weights = _weights(sample_weight, len(y))
        started = time.perf_counter()
        params = dict(GA_V14_XGB_PARAMS, random_state=self.seed, n_jobs=self.n_jobs, device='cpu')
        self.model = xgboost.XGBRegressor(**params)
        self.model.fit(x, y, sample_weight=weights)
        return dict(fit_seconds=time.perf_counter()-started, train_count=len(y), params=params,
            xgboost_version=xgboost.__version__, label_source='direct observed docking utility; no distillation',
            weighting='uniform' if sample_weight is None else 'explicit acquired-row sample weights')

    def fit(self, smiles, y, sample_weight=None):
        return self.fit_features(product_morgan(smiles, n_jobs=self.n_jobs), y, sample_weight)

    def predict_features(self, features, batch_size=50000):
        if self.model is None:
            raise RuntimeError('Call fit first')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        x = self._features(features)
        if not len(x):
            return np.empty(0, np.float32)
        return np.concatenate([self.model.predict(x[start:start+batch_size])
                               for start in range(0, len(x), batch_size)]).astype(np.float32)

    def predict(self, smiles):
        return self.predict_features(product_morgan(smiles, n_jobs=self.n_jobs))


def _weights(weights, count):
    values = np.ones(count, np.float32) if weights is None else np.asarray(weights, np.float32)
    if values.shape != (count,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Weights must be finite positive values for every acquired row')
    return values


class _ReactionFactorNetwork(_FactorNetwork):
    def __init__(self, input_dim, n_reactions, rank, hidden, additive):
        super().__init__(input_dim+n_reactions, n_reactions, rank, hidden, additive)
        self.n_reactions = n_reactions

    def components(self, x):
        # Typed synthon reaction IDs occupy the final input column. Their one-hot
        # block has unit scale, never the .001 floor of numerical descriptors.
        reaction = nn.functional.one_hot(x[:, -1].long(), self.n_reactions).float()
        return super().components(torch.cat([x[:, :-1], reaction], dim=1))


class WeightedFactorRegressor(FactorRegressor):
    """Original CP by default; explicit weighted and reaction-conditioned variants."""
    def __init__(self, *args, synthon_reaction_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthon_reaction_ids = None
        if synthon_reaction_ids is not None:
            ids = np.asarray(synthon_reaction_ids, np.int64)
            if ids.shape != (len(self.features),) or np.any(ids < 0) or np.any(ids >= self.n_reactions):
                raise ValueError('Need a valid reaction ID for each typed synthon row')
            self.synthon_reaction_ids = ids

    def fit(self, tuples, reaction_ids, y, epochs=120, patience=15, batch_size=512, sample_weight=None):
        if sample_weight is None and self.synthon_reaction_ids is None:
            result = super().fit(tuples, reaction_ids, y, epochs, patience, batch_size)
            return dict(result, weighting='uniform', reaction_conditioning=False,
                        label_source='direct observed docking utility; no distillation')
        rows, reactions = self._inputs(tuples, reaction_ids)
        y = np.asarray(y, np.float32); weights = _weights(sample_weight, len(y))
        if y.shape != (len(rows),) or len(y) < 10 or not np.isfinite(y).all():
            raise ValueError('Need at least ten finite acquired utilities')
        if min(epochs, patience, batch_size) < 1:
            raise ValueError('Training counts must be positive')
        started = time.perf_counter(); torch.set_num_threads(4); torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed+7919); order = rng.permutation(len(y))
        cut = max(2, int(len(y)*.2)); val, train = order[:cut], order[cut:]
        used = np.unique(rows[train][rows[train] >= 0])
        self.x_mean = self.features[used].mean(0); self.x_scale = np.maximum(self.features[used].std(0),1e-3)
        self.x = torch.tensor(np.clip((self.features-self.x_mean)/self.x_scale,-8,8),device=self.device)
        if self.synthon_reaction_ids is not None:
            ids = torch.tensor(self.synthon_reaction_ids,device=self.device,dtype=torch.float32)
            self.x = torch.cat([self.x,ids[:,None]],dim=1)
        self.y_mean = float(y[train].mean()); self.y_scale = max(float(y[train].std()),1e-5)
        network = _FactorNetwork if self.synthon_reaction_ids is None else _ReactionFactorNetwork
        self.model = network(self.features.shape[1],self.n_reactions,self.rank,self.hidden,self.additive).to(self.device)
        rr,ri = torch.tensor(rows,device=self.device),torch.tensor(reactions,device=self.device)
        target = torch.tensor((y-self.y_mean)/self.y_scale,device=self.device)
        ww = torch.tensor(weights,device=self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=.002,weight_decay=1e-4)
        best,stale,selected,history = float('inf'),0,0,[]
        for epoch in range(int(epochs)):
            self.model.train()
            for ix in np.array_split(rng.permutation(train),max(1,math.ceil(len(train)/batch_size))):
                pred = self.model(self.x,rr[ix],ri[ix]); loss = ((pred-target[ix]).square()*ww[ix]).sum()/ww[ix].sum()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite weighted CP loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),10.);optimizer.step()
            self.model.eval()
            with torch.no_grad():
                error_sum,weight_sum=0.,0.
                for ix in np.array_split(val,max(1,math.ceil(len(val)/batch_size))):
                    pred=self.model(self.x,rr[ix],ri[ix]);error_sum+=float(((pred-target[ix]).square()*ww[ix]).sum());weight_sum+=float(ww[ix].sum())
                error=error_sum/weight_sum
            history.append(error)
            if error < best:
                best,stale,selected=error,0,epoch+1;state=copy.deepcopy(self.model.state_dict())
            else: stale+=1
            if stale >= patience: break
        self.model.load_state_dict(state);self.model.eval();self.refresh_cache()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        return dict(epochs=epoch+1,selected_epoch=selected,best_val_nmse=best,
            fit_seconds=time.perf_counter()-started,train_count=len(train),validation_count=len(val),
            validation_history=history,rank=self.rank,additive=self.additive,
            weighting='uniform' if sample_weight is None else 'explicit weighted MSE in fit and acquired-only validation',
            reaction_conditioning=self.synthon_reaction_ids is not None,
            reaction_indicator_scale='one-hot unit scale; excluded from descriptor standardization',
            label_source='direct observed docking utility; no distillation')
