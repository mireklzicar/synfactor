# OPDiv100 bounds

Utility is negative mean docking score; larger is better. Every portfolio contains 100 molecules with strict Morgan Tc < 0.30. Feasible bounds are not automatically exact optima.

Raw and portfolio reference policies are separate runs, each with its own 10,000 charged-call budget. The optional development continuation inherits 4,000 portfolio calls and may add 6,000 new calls; it is not an independent replicate. TYK2 seed 0 is development evidence, not an independent confirmation.

## Latest common evaluated budget: 10,000 calls

| Method | Feasible lower bound | Global upper bound | Status | Certified |
|---|---:|---:|---|---|
| SynFactor + product XGB: raw | 7.457220 | 7.457220 | optimal | Yes |
| SynFactor + product XGB: portfolio | 8.980480 | 8.980480 | optimal | Yes |
| GA-DCSO v14 | 8.424790 | 8.424790 | optimal | Yes |
| SpaceHASTEN fast | 7.212630 | 7.212630 | optimal | Yes |
| SpaceHASTEN accurate | 7.312440 | 7.377174 | bounded | No |
| SynFactor v1 | 7.932450 | 7.932450 | optimal | Yes |

## Final 10,000-call bounds

| Method | Feasible lower bound | Global upper bound | Status | Certified |
|---|---:|---:|---|---|
| SynFactor + product XGB: raw | 7.457220 | 7.457220 | optimal | Yes |
| SynFactor + product XGB: portfolio | 8.980480 | 8.980480 | optimal | Yes |
| GA-DCSO v14 | 8.424790 | 8.424790 | optimal | Yes |
| SpaceHASTEN fast | 7.212630 | 7.212630 | optimal | Yes |
| SpaceHASTEN accurate | 7.312440 | 7.377174 | bounded | No |
| SynFactor v1 | 7.932450 | 7.932450 | optimal | Yes |
| Explore/exploit + CP fibers (4K inherited) | 8.379350 | 8.379350 | optimal | Yes |

## Development continuation

Latest matched prefix: 10,000 charged calls, including the inherited 4,000. See development_branch_comparison.csv for all seven series at each available matched branch budget.
Continuation OPDiv100 interval: [8.379349999999999, 8.379350000000004]; status: optimal. This branch must not be counted as a third independent replicate.

Repeated baseline evaluations must agree on raw/GPDiv metrics. Their valid full-domain OPDiv intervals are intersected (largest feasible lower bound, smallest upper bound); no restricted-domain bound is used.
