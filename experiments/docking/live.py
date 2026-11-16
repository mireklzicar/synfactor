"""Local GPU SynFactor search with the isolated remote Dock-12 oracle.

Every round scores the complete Cartesian release, retains a bounded pool,
mixes exploitation/information/random acquisition, and charges all new routes.
The shared 1K warm cohort is charged. Oracle work and search have separate
durable checkpoints; uncertain remote submissions are never blindly retried.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from .cartesian import stream_topk
from .factor_model import FactorRegressor
from .fullspace import FullSpace
from .information import LinearInformation
from .run import diverse_select


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def protocol_digest(protocol):
    identity={k:v for k,v in protocol.items() if k not in ("free_disk_gib","preflight_only")}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def validate_state(state, warm):
    rounds,history=state["rounds"],state["history"]
    if len(rounds)>9 or len(history)!=1000+1000*len(rounds) or history[:1000]!=warm:
        raise ValueError("Checkpoint no longer matches its charged warm/batch ledger")
    if len({r["product_id"] for r in history})!=len(history):
        raise ValueError("Duplicate charged identities in checkpoint")
    for iteration,record in enumerate(rounds,1):
        if record["iteration"]!=iteration or record["calls"]!=1000*(iteration+1):
            raise ValueError("Nonconsecutive checkpoint rounds")
        if any(r["iteration"]!=iteration for r in history[1000*iteration:1000*(iteration+1)]):
            raise ValueError("Checkpoint observations have wrong batch identity")


class RemoteOracle:
    def __init__(self, host, key, python, root):
        options = ["-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                   "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
        self.ssh, self.scp = ["ssh", *options, host], ["scp", *options]
        self.host, self.python, self.root = host, python, root
        self.script = root+"/remote_oracle.py"

    def command(self, arguments, **kwargs):
        environment = ["env", "PATH=/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"]
        return subprocess.run([*self.ssh, shlex.join(environment+arguments)], check=True, **kwargs)

    def preflight(self, target, seed):
        result = self.command([self.python, self.script, "--target", target, "--seed", str(seed)],
                              capture_output=True, text=True)
        return json.loads(result.stdout)

    def query(self, proposal, folder):
        request = folder/"proposal.json"
        if request.exists() and json.loads(request.read_text()) != proposal:
            raise ValueError("Previously committed proposal changed")
        write(request, proposal)
        response = folder/"normalized.json"
        expected = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
        if not response.exists():
            remote_request = self.root+f"/request_{proposal['run_id']}_{proposal['batch_id']}.json"
            subprocess.run([*self.scp, str(request), f"{self.host}:{remote_request}"], check=True)
            with (folder/"remote.log").open("a") as log:
                self.command([self.python, self.script, "--proposal", remote_request, "--execute"],
                             stdout=log, stderr=subprocess.STDOUT)
            remote_result = self.root+f"/{proposal['run_id']}/{proposal['batch_id']}/normalized.json"
            subprocess.run([*self.scp, f"{self.host}:{remote_result}", str(response)], check=True)
        result = json.loads(response.read_text())
        if result["request_sha256"] != expected or result.get("smoke_only"):
            raise ValueError("Remote response does not match the charged adaptive request")
        if [r["product_id"] for r in result["rows"]] != proposal["product_ids"]:
            raise ValueError("Remote oracle changed route identity/order")
        return result


def warm_start(repo, target, seed, space):
    cohort = repo/f"artifacts/seeds/freedom_296b/seed_{seed}/selected_10k/subselections"
    ids_path = cohort/"budget_01000.ids"
    matrix = repo/f"artifacts/dock12_seed_screen/freedom/seed_{seed}/results_long.csv"
    ids = ids_path.read_text().splitlines()
    candidates = pd.read_csv(cohort/"budget_01000_candidates.csv").set_index("product_id")
    wanted = set(ids)
    with matrix.open() as handle:
        scores = {r["product_id"]:r for r in csv.DictReader(handle)
                  if r["target_id"] == target and r["product_id"] in wanted}
    if len(ids) != 1000 or len(wanted) != 1000 or set(scores) != wanted:
        raise ValueError("Invalid/missing shared warm cohort")
    rows = []
    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.DictWriter(csv_buffer,fieldnames=["product_id","status","score"])
    csv_writer.writeheader()
    for pid in ids:
        r = scores[pid]
        valid = str(r["valid"]).lower() in ("true", "1", "yes")
        csv_writer.writerow(dict(product_id=pid,status="valid" if valid else "failed",score=r["score"] if valid else ""))
        value = float(r["score"]) if valid else None
        if value is not None and (not np.isfinite(value) or abs(value) >= 100):
            valid, value = False, None
        rows.append(dict(product_id=pid, smiles=str(candidates.loc[pid,"smiles"]),
                         score=value, status="valid" if valid else "failed", iteration=-1,
                         source="charged_shared_warm", charged=True))
    # Verify typed identity before any new oracle work.
    space.decode(ids)
    return rows, dict(ids_sha256=digest(ids_path), matrix_sha256=digest(matrix),
                      csv_sha256=hashlib.sha256(csv_buffer.getvalue().encode()).hexdigest(),
                      candidates_sha256=digest(cohort/"budget_01000_candidates.csv"))


def propose(space, history, config, iteration):
    rows, rids = space.decode([r["product_id"] for r in history])
    y = np.asarray([-float(r["score"]) if r["status"] == "valid" and r["score"] is not None else 0.
                    for r in history], np.float32)
    t0 = time.perf_counter()
    model = FactorRegressor(space.features, len(space.reactions), rank=64, additive=True,
                            device="cuda", seed=config["seed"]+iteration*17)
    fit = model.fit(rows, rids, y, epochs=180)
    training_predictions = model.predict(rows, rids)
    sweep = stream_topk(model, space.grids, k=20000, excluded_tuples=rows,
                        excluded_reaction_ids=rids, block_rows=4096, block_cols=8192,
                        max_seconds=config["sweep_seconds"])
    top_rows, top_rids, means = sweep["tuples"], sweep["reaction_ids"], sweep["scores"]
    n_information = int(round(1000*config["information_fraction"]))
    exploit_count = 900-n_information
    references = rows[np.argsort(-y, kind="stable")[:100]]
    # Selection accepts a local tuple array. Its reference rows are appended;
    # descriptors remain the full synthon cache and are gathered only on demand.
    diversity_rows = np.concatenate([top_rows, references])
    chosen = diverse_select(np.arange(len(top_rows)), means, exploit_count, None, None,
                            np.arange(len(top_rows),len(diversity_rows)), .5, "cuda",
                            synthon_rows=diversity_rows, synthon_features=space.features)
    acquired_rows, acquired_rids = top_rows[chosen], top_rids[chosen]
    seen = {(int(r),*map(int,row)) for row,r in zip(rows,rids)}
    seen.update((int(r),*map(int,row)) for row,r in zip(acquired_rows,acquired_rids))
    information = None
    if n_information:
        rng = np.random.default_rng(config["seed"]+7919*iteration)
        random_rows, random_rids = space.sample(10000, rng, seen)
        eligible = np.ones(len(top_rows),bool);eligible[chosen]=False
        elite_rows, elite_rids = top_rows[eligible][:10000], top_rids[eligible][:10000]
        pool_rows = np.concatenate([elite_rows,random_rows])
        pool_rids = np.concatenate([elite_rids,random_rids])
        # Distinct route identities, preserving pool order.
        combined = np.column_stack([pool_rids,pool_rows])
        _, unique = np.unique(combined, axis=0, return_index=True)
        unique.sort();pool_rows=pool_rows[unique];pool_rids=pool_rids[unique]
        info = LinearInformation(model.acquisition_features(rows,rids), y, training_predictions, device="cuda")
        positions, information = info.select(
            model.acquisition_features(pool_rows,pool_rids), model.predict(pool_rows,pool_rids), n_information,
            condition_features=model.acquisition_features(acquired_rows,acquired_rids), quality_weight=.25)
        acquired_rows = np.concatenate([acquired_rows,pool_rows[positions]])
        acquired_rids = np.concatenate([acquired_rids,pool_rids[positions]])
        seen.update((int(r),*map(int,row)) for row,r in zip(pool_rows[positions],pool_rids[positions]))
    random_rows, random_rids = space.sample(100,np.random.default_rng(config["seed"]+991+iteration),seen)
    acquired_rows=np.concatenate([acquired_rows,random_rows]);acquired_rids=np.concatenate([acquired_rids,random_rids])
    ids=space.ids(acquired_rows,acquired_rids)
    if len(ids)!=1000 or len(set(ids))!=1000 or set(ids)&{r["product_id"] for r in history}:
        raise ValueError("Acquisition violated the unique charged-route budget")
    diagnostics=dict(fit=fit,sweep=sweep["diagnostics"],information=information,
                     search_seconds=time.perf_counter()-t0, exploit_count=exploit_count,
                     information_count=n_information, random_count=100,
                     predictions=model.predict(acquired_rows,acquired_rids).tolist())
    del model
    torch.cuda.empty_cache()
    return ids,diagnostics


def run_cell(args, space, oracle, target):
    run_id=f"sf_freedom_seed{args.seed}_{target}_info{round(args.information_fraction*100)}"
    root=args.out/run_id;root.mkdir(parents=True,exist_ok=True)
    with (root/"cell.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        warm,warm_hashes=warm_start(args.benchmark_repo,target,args.seed,space)
        protocol=oracle.preflight(target,args.seed)
        if protocol["warm_ordered_ids_sha256"]!=warm_hashes["ids_sha256"]:
            raise ValueError("Remote warm cohort differs from local charged initialization")
        for name,sha in space.manifest["source_sha256"].items():
            if protocol["source_sha256"].get("/home/mirekl/TWIST_Navigator/freedom5_skach/"+name)!=sha:
                raise ValueError("Remote and GPU-local chemical spaces differ")
        local_matrix=args.benchmark_repo/f"artifacts/dock12_seed_screen/freedom/seed_{args.seed}/results_long.csv"
        if protocol["source_sha256"].get(str(local_matrix))!=warm_hashes["matrix_sha256"]:
            raise ValueError("Local and remote warm-score matrices differ")
        config=dict(target=target,seed=args.seed,information_fraction=args.information_fraction,
                    sweep_seconds=args.sweep_seconds,budget=10000,warm_charged=1000,
                    batch_size=1000,rank=64,additive=True,epochs=180,diversity=.5,
                    diversity_features="capped-synthon Morgan512 OR", information_quality=.25,
                    space_sha256=space.manifest["source_sha256"],products=space.manifest["products"],
                    warm=warm_hashes,grid_sha256=protocol["grid_sha256"],
                    oracle_protocol_sha256=protocol_digest(protocol),max_cell_seconds=args.max_cell_seconds,
                    code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name("factor_model.py"),
                        Path(__file__).with_name("cartesian.py"),Path(__file__).with_name("information.py"),
                        Path(__file__).with_name("fullspace.py"),Path(__file__).with_name("run.py")]})
        if (root/"config.json").exists() and json.loads((root/"config.json").read_text())!=config:
            raise ValueError("Cell configuration/source changed; refusing unsafe resume")
        write(root/"config.json",config);write(root/"remote_preflight.json",protocol)
        if (root/"CELL_DONE").exists():
            return
        state=json.loads((root/"state.json").read_text()) if (root/"state.json").exists() else dict(history=list(warm),rounds=[])
        validate_state(state,warm)
        write(root/"state.json",state)
        for iteration in range(1+len(state["rounds"]),10):
            consumed=sum(r["search"]["search_seconds"]+r["oracle_seconds"] for r in state["rounds"])
            if consumed>=args.max_cell_seconds:
                raise TimeoutError("Cell reached its wall-time budget; completed calls/checkpoints retained")
            batch=root/f"batch_{iteration:02d}";batch.mkdir(exist_ok=True)
            if (batch/"proposal.json").exists():
                proposal=json.loads((batch/"proposal.json").read_text())
                diagnostics=json.loads((batch/"search.json").read_text())
            else:
                ids,diagnostics=propose(space,state["history"],config,iteration)
                proposal=dict(target=target,seed=args.seed,run_id=run_id,batch_id=batch.name,product_ids=ids)
                write(batch/"search.json",diagnostics);write(batch/"proposal.json",proposal)
            ids=proposal["product_ids"]
            if len(ids)!=1000 or len(set(ids))!=1000 or set(ids)&{r["product_id"] for r in state["history"]}:
                raise ValueError("Cached proposal violates the charged route budget")
            if any(proposal[k]!=v for k,v in dict(target=target,seed=args.seed,run_id=run_id,batch_id=batch.name).items()):
                raise ValueError("Cached proposal belongs to another cell/batch")
            space.decode(ids)
            print(f"{target} round{iteration}: GPU proposal ready; submitting1000 charged routes",flush=True)
            result=oracle.query(proposal,batch)
            if protocol_digest(result["protocol"])!=config["oracle_protocol_sha256"]:
                raise ValueError("Returned oracle protocol differs from pinned preflight")
            state["history"].extend(dict(product_id=r["product_id"],smiles=r["smiles"],score=r["score"],
                status="valid" if r["valid"] else "failed",iteration=iteration,source="live_glide",charged=True)
                for r in result["rows"])
            if len({r["product_id"] for r in state["history"]})!=len(state["history"]):
                raise ValueError("Oracle ledger contains duplicate routes")
            state["rounds"].append(dict(iteration=iteration,calls=len(state["history"]),
                                       search=diagnostics,oracle_seconds=result["seconds"]))
            validate_state(state,warm)
            write(root/"state.json",state)
            valid=[r["score"] for r in state["history"] if r["status"]=="valid"]
            print(f"{target} calls={len(state['history'])} best={min(valid):.4f} valid={len(valid)}",flush=True)
        if len(state["history"])!=10000:
            raise ValueError("Cell did not complete its charged budget")
        write(root/"summary.json",dict(target=target,seed=args.seed,calls=10000,
              unique_product_ids=len({r["product_id"] for r in state["history"]}),
              search_seconds=sum(r["search"]["search_seconds"] for r in state["rounds"]),
              oracle_seconds=sum(r["oracle_seconds"] for r in state["rounds"]),
              grid_sha256=config["grid_sha256"],warm_charged=1000,new_attempts=9000))
        (root/"CELL_DONE").write_text("10000 charged routes completed\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--benchmark-repo",type=Path,default=Path("/home/mirekl/SynthonBench_ICML/synthonbench_v2"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--targets",nargs="+")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--information-fraction",type=float,default=.2)
    p.add_argument("--sweep-seconds",type=float,default=300)
    p.add_argument("--max-cell-seconds",type=float,default=3600,
                   help="Soft cap checked between completed batches; no interruption of in-flight docking")
    p.add_argument("--host",required=True)
    p.add_argument("--key",type=Path,required=True)
    p.add_argument("--remote-python",default="/home/ubuntu/dmc-navigator-prod-v040/.venv/bin/python")
    p.add_argument("--remote-root",default="/home/ubuntu/synfactor_dock12_20261116")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not 0<=a.information_fraction<.9 or a.seed not in (0,1,2):
        p.error("Invalid information fraction/seed")
    a.out.mkdir(parents=True,exist_ok=True)
    roster=json.loads((a.benchmark_repo/"configs/targets_dock12.json").read_text())["targets"]
    targets=a.targets or [t["target_id"] for t in roster]
    if len(set(targets))!=len(targets) or set(targets)-{t["target_id"] for t in roster}:
        p.error("Unknown/duplicate target")
    oracle=RemoteOracle(a.host,a.key,a.remote_python,a.remote_root)
    if not a.execute:
        for target in targets:
            write(a.out/f"preflight_{target}.json",oracle.preflight(target,a.seed))
        print(f"Read-only preflight completed for {len(targets)} targets")
        return
    torch.set_num_threads(4)
    space=FullSpace(a.cache)
    with (a.out/"campaign.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for target in targets:
            run_cell(a,space,oracle,target)


if __name__=="__main__":
    main()
