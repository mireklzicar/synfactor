"""Small offline demo teacher, plus pinned OpenADMET/CheMeleon adapters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

REVISIONS = {
    "herg-chemeleon-baseline": "5c4fb081dfec62a5108423d0f86ec99023551bd8",
    "permeability-logd-ppb-chemeleon-baseline": "6fb7782db474a9c9198d4ace3a5c6b6a6abb304b",
    "cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1": "ef24cf941ae21c7d7a64df378a846bd2066eceda",
    "microsomal-clearance-chemeleon-v1": "e13549384674e70a536097b7175932e36d5ff271",
    "pxr-chemeleon-v1": "6273778544ccbac389456ca0cbf802d3e26ec949",
    "logd-chemprop_foundation_minimol-v1": "d9007fdfb18c0ceda4471207ea4dcdd5d071d65b",
    "logd-chemprop_foundation_surrogate-v1": "6fd9462dc1db1d094d0ddd5a678321658fc772d6",
    "mpbp-chemprop_foundation_minimol-v1": "5d081db6f316f78c4ad97a30c554288833ac6187",
    "mpbp-chemprop_foundation_surrogate-v1": "009eea9b735edc42cf86608620ac87d2b543ce44",
}


class RDKitTeacher:
    """Morgan counts + cheap descriptors -> full-product Crippen logP target.

    This is an educational distillation task, not an ADMET or hERG model.
    """

    targets = ["RDKit_CrippenLogP"]
    provenance = {
        "teacher": "RDKit CrippenLogP",
        "embedding": "Morgan counts (512) + 4 descriptors",
    }

    def encode(self, smiles):
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        rows = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fp = gen.GetCountFingerprintAsNumPy(mol)
            extra = [
                Crippen.MolLogP(mol),
                Descriptors.MolWt(mol),
                Descriptors.TPSA(mol),
                mol.GetNumHeavyAtoms(),
            ]
            rows.append(np.concatenate([fp, extra]))
        return np.asarray(rows, dtype=np.float32)

    def predict(self, smiles):
        return np.asarray(
            [[Crippen.MolLogP(Chem.MolFromSmiles(s))] for s in smiles], dtype=np.float32
        )


class OpenADMETTeacher:
    """Frozen post-message-passing embeddings and original property head.

    Accepts the nine pinned model repositories or a local anvil_training folder.
    No weights are distributed by SynFactor. Downloads retain upstream licenses.
    """

    def __init__(
        self,
        repository="herg-chemeleon-baseline",
        device="cpu",
        model_dir=None,
        foundation_root=None,
        batch_size=256,
    ):
        from huggingface_hub import snapshot_download
        from openadmet.models.architecture.chemprop import ChemPropModel
        from openadmet.models.features.chemprop import ChemPropFeaturizer

        if model_dir is None:
            if repository not in REVISIONS:
                raise ValueError(
                    "Unknown repository; use --model-dir for another compatible model"
                )
            root = Path(
                snapshot_download(
                    f"openadmet/{repository}", revision=REVISIONS[repository]
                )
            )
        else:
            root = Path(model_dir)
        directory = (
            root / "anvil_training" if (root / "anvil_training").is_dir() else root
        )
        parameters = json.loads((directory / "model.json").read_text())
        foundation = parameters.get("from_foundation")
        if foundation and foundation != "chemeleon":
            source = Path(foundation)
            candidates = [source] if source.is_absolute() else [directory / source]
            if foundation_root:
                candidates.insert(0, Path(foundation_root) / source)
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                downloaded = Path(
                    snapshot_download(
                        "openadmet/chemprop-foundation-pretraining-weights",
                        revision="0f8201d06c4656c3f97bc244ce2dc18ac2881be9",
                    )
                )
                resolved = downloaded / source.name.replace("_mp.pt", ".pt")
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Missing foundation {foundation}; supply --foundation-root"
                )
            parameters["from_foundation"] = str(resolved.resolve())
        wrapper = ChemPropModel(**parameters)
        wrapper.build()
        wrapper.load(directory / "model.pth")
        self.estimator = wrapper.estimator.to(device).eval()
        self.featurizer = ChemPropFeaturizer(
            n_jobs=0, batch_size=batch_size, normalize_targets=True, shuffle=False
        )
        self.device = device
        # Obtain target order from the model's own recipe, never from alphabetic sorting.
        recipe = directory / "recipe_components" / "data.yaml"
        self.targets = None
        if recipe.is_file():
            import yaml

            config = yaml.safe_load(recipe.read_text())
            self.targets = config.get("target_cols")
        if not self.targets:
            self.targets = [
                f"{repository}:output_{i}" for i in range(parameters["n_tasks"])
            ]
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        self.provenance = {
            "teacher": f"openadmet/{repository}",
            "revision": REVISIONS.get(repository)
            if model_dir is None
            else "local checkpoint",
            "checkpoint_sha256": hashlib.sha256(
                (directory / "model.pth").read_bytes()
            ).hexdigest(),
            "embedding": "frozen post-message-passing fingerprint",
        }

    @torch.inference_mode()
    def _run(self, smiles, embeddings):
        output = []
        for start in range(0, len(smiles), 10000):
            block = list(smiles[start : start + 10000])
            loader, *_ = self.featurizer.featurize(pd.Series(block))
            for bmg, vertex, extra, *_ in loader:
                bmg.to(self.device)
                vertex = None if vertex is None else vertex.to(self.device)
                extra = None if extra is None else extra.to(self.device)
                latent = self.estimator.fingerprint(bmg, vertex, extra)
                values = latent if embeddings else self.estimator.predictor(latent)
                output.append(values.float().cpu().numpy())
        result = np.concatenate(output)
        if len(result) != len(smiles) or not np.isfinite(result).all():
            raise ValueError("Teacher returned missing rows or nonfinite values")
        return result

    def encode(self, smiles):
        return self._run(smiles, True)

    def predict(self, smiles):
        return self._run(smiles, False)
