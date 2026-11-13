# Example data

Copied unchanged from [SynthonOR](https://github.com/mireklzicar/synthonor/tree/094eea2146fa9a1830621700512128647e5ebdb8), commit `094eea2146fa9a1830621700512128647e5ebdb8`.

The synthon table is the same bundled `synthon_space_1M.tsv`; reaction SMARTS come from `src/synthonor/data/synthon_space_1M.reactions.tsv`. This example has 6,273 component-slot records and 42 reactions, representing 990,829 row combinations (before molecular identity deduplication). It is a small example slice, not the full commercial chemical library.

SynthonOR is MIT licensed, copyright (c) 2026 Deep MedChem; that notice is retained in the root LICENSE. The figure and research metrics were supplied by the SynFactor author. Downloaded upstream model weights retain their own model-card licenses.

Columns: `smiles`, `synthon_id`, one-based `position`, `reaction_id`. U/Np/Pu are connector markers. Only combinations within a reaction and its declared slots are valid.

SHA-256:

- `synthon_space_1M.tsv`: `d234a99d2695ed5e5269fe162bba12b99be1cd9c8d5ec0c43e57fa4180aee841`
- `reactions.tsv`: `a7d877109689fb678a346a5b4166199b772d15621e41c99cae0c80c86b2249cb`
