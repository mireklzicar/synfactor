"""Durable local evaluation/cleanup supervisor; never submits oracle queries.

Run separately from the immutable live campaign. Completed cells are evaluated
serially. Instance restoration is opt-in, ownership-bound, and fails closed on
remote uncertainty. This program never restarts the campaign or a docking batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

METHODS = {"synfactor", "ga_dcso_v14_screening", "spacehasten_fast", "spacehasten_accurate"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def process_identity(pid, proc=Path("/proc")):
    """Boot ID + kernel start ticks protect against PID reuse and host reboot."""
    try:
        text = (proc / str(pid) / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        command = (proc / str(pid) / "cmdline").read_bytes()
        return dict(pid=pid, start_ticks=int(fields[19]),
                    boot_id=(proc / "sys/kernel/random/boot_id").read_text().strip(),
                    cmdline_sha256=hashlib.sha256(command).hexdigest())
    except FileNotFoundError:
        return None


def process_state(expected):
    actual = process_identity(expected["pid"])
    return "gone" if actual is None else ("running" if actual == expected else "pid_reused")


def bind_process(pid, campaign):
    identity = process_identity(pid)
    if identity is None:
        raise ValueError("Campaign already absent; a previously pinned monitor state is required")
    argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
    if "experiments.docking.live" not in argv or "--out" not in argv:
        raise ValueError("PID is not the requested live campaign")
    if Path(argv[argv.index("--out")+1]).resolve() != campaign:
        raise ValueError("Campaign PID output directory differs")
    return identity


def validate_cell(cell):
    config, preflight = read(cell / "config.json"), read(cell / "remote_preflight.json")
    state = read(cell / "state.json")
    history = state["history"]
    if not (cell / "CELL_DONE").exists() or len(history) != 10000:
        raise ValueError("Evaluation requires a complete 10,000-call cell")
    if len({r["product_id"] for r in history}) != 10000:
        raise ValueError("Charged product IDs are not unique")
    if sum(int(r["iteration"]) < 0 for r in history) != 1000:
        raise ValueError("Exactly 1000 charged warm rows required")
    if config["budget"] != 10000 or config["warm_charged"] != 1000:
        raise ValueError("Unexpected budget configuration")
    if config["grid_sha256"] != preflight["grid_sha256"]:
        raise ValueError("Grid differs from remote preflight")
    if config["warm"]["ids_sha256"] != preflight["warm_ordered_ids_sha256"]:
        raise ValueError("Warm identity differs from remote preflight")
    if config["target"] != preflight["target"] or config["seed"] != preflight["seed"]:
        raise ValueError("Cell target/seed differs from remote preflight")
    return config


def evaluation_complete(directory, history_sha, config):
    if not (directory / "comparison.csv").exists():
        return False
    summary = read(directory / "summary.json")
    return (summary.get("comparison_verified") is True
            and summary.get("history_sha256") == history_sha
            and summary.get("grid_sha256") == config["grid_sha256"]
            and summary.get("warm_sha256") == config["warm"]["csv_sha256"]
            and {c["method"] for c in summary.get("cells", [])} == METHODS
            and all(set(c["domains"]) == {"all-valid", "druglike-v1"} for c in summary["cells"]))


def portfolio_comparisons(summary):
    output = []
    methods = {c["method"]: c for c in summary["cells"]}
    for domain in ("all-valid", "druglike-v1"):
        cp = next(p for p in methods["synfactor"]["domains"][domain]["portfolios"] if p["metric"] == "OPDiv")
        for method in sorted(METHODS - {"synfactor"}):
            baseline = next(p for p in methods[method]["domains"][domain]["portfolios"] if p["metric"] == "OPDiv")
            lo, hi = cp.get("feasible_lower_bound"), cp.get("upper_bound")
            blo, bhi = baseline.get("feasible_lower_bound"), baseline.get("upper_bound")
            claim = "not_separated_by_bounds"
            if lo is not None and bhi is not None and lo > bhi + 1e-6:
                claim = "certified_synfactor_win"
            elif blo is not None and hi is not None and blo > hi + 1e-6:
                claim = "certified_baseline_win"
            output.append(dict(domain=domain, baseline=method, synfactor_lower=lo,
                               synfactor_upper=hi, baseline_lower=blo, baseline_upper=bhi, claim=claim))
    return output


REMOTE_IDLE = r'''
import os,json,pathlib,subprocess,time,collections
blockers=[]; me=os.getpid(); ancestors=set(); parent=me
while parent>1:
 ancestors.add(parent)
 try:
  text=pathlib.Path('/proc',str(parent),'stat').read_text();parent=int(text[text.rfind(')')+2:].split()[1])
 except (OSError,ValueError):break
def check(command):
 r=subprocess.run(command,capture_output=True,text=True,timeout=15)
 if r.returncode:raise RuntimeError('inspection command failed: '+command[0])
 return r.stdout
try:
 containers=check(['docker','ps','--format','{{.Names}}']).splitlines()
 if containers:blockers.append({'kind':'active_containers','count':len(containers)})
 units=json.loads(check(['systemctl','--user','list-units','--type=service','--state=running','--output=json','--no-pager']))
 active=[r['unit'] for r in units if any(k in r['unit'].lower() for k in ['synthon','spacehasten','navigator','campaign','docking','molpal'])]
 if active:blockers.append({'kind':'active_campaign_services','units':active})
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   pid=int(p.name)
   if pid in ancestors:continue
   cmd=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if not cmd[0]:continue
   comm=(p/'comm').read_text().strip();uid=p.stat().st_uid
   joined=' '.join(cmd).lower()
   if any(x in joined for x in ['remote_oracle.py','dock_coupled','run_batch.py','/glide','/ligprep','/epik','run_public_roster','run_spacehasten','run_navigator']):
    blockers.append({'kind':'oracle_or_campaign_process','pid':pid,'comm':comm});continue
   if uid>=1000 and comm not in ['systemd','(sd-pam)','dbus-daemon','ssh-agent','gpg-agent']:
    blockers.append({'kind':'other_user_process','pid':pid,'comm':comm});continue
   if uid==0 and any(comm.startswith(x) for x in ['python','java','julia','Rscript','node','mpirun']):
    if not any(x in joined for x in ['/usr/bin/networkd-dispatcher','unattended-upgrade-shutdown']):
     blockers.append({'kind':'other_root_compute','pid':pid,'comm':comm})
  except FileNotFoundError:pass
  except PermissionError:blockers.append({'kind':'process_inspection_denied','pid':int(p.name)})
 def cpu():
  fields=list(map(int,pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(fields),fields[3]+fields[4]
 a=cpu();time.sleep(.5);b=cpu();busy=1-(b[1]-a[1])/max(1,b[0]-a[0]);cores=busy*(os.cpu_count() or 1)
 if cores>2:blockers.append({'kind':'unexplained_cpu_activity','busy_cores':round(cores,2)})
 print(json.dumps({'inspection_complete':True,'idle':not blockers,'blockers':blockers,'busy_cores':round(cores,2)}))
except Exception as error:
 print(json.dumps({'inspection_complete':False,'idle':False,'blockers':[{'kind':'inspection_failed','error_type':type(error).__name__}]}))
'''


def remote_idle(host, key):
    result = subprocess.run(["ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=10", host, "python3", "-"],
                            input=REMOTE_IDLE, text=True, capture_output=True, timeout=45)
    if result.returncode:
        return dict(inspection_complete=False, idle=False, blockers=[dict(kind="ssh_inspection_failed")])
    try:
        return json.loads(result.stdout)
    except ValueError:
        return dict(inspection_complete=False, idle=False, blockers=[dict(kind="unparseable_remote_inspection")])


def ownership(path):
    record = read(path)
    required = ("instance_id", "profile", "region", "public_ip")
    if (any(not isinstance(record.get(k), str) or not record[k].strip() for k in required)
            or not record["instance_id"].startswith("i-")
            or record.get("previous_state") != "stopped"
            or record.get("start_requested_by_this_campaign") is not True):
        raise ValueError("Missing matching originally-stopped instance ownership")
    return record


def aws(record, *arguments):
    result = subprocess.run(["aws", "--profile", record["profile"], "--region", record["region"], "ec2", *arguments,
                             "--output", "json"], text=True, capture_output=True, timeout=30, check=True)
    return json.loads(result.stdout)


def restore_instance(args, monitor):
    if sha(args.instance_audit) != monitor["binding"]["instance_audit_sha256"]:
        raise ValueError("Owned-start audit changed since supervisor launch")
    record = ownership(args.instance_audit)
    if process_state(monitor["process"]) == "running":
        raise ValueError("Refusing instance shutdown while campaign remains alive")
    instance_id = record["instance_id"]
    result = aws(record, "describe-instances", "--instance-ids", instance_id)
    instance = result["Reservations"][0]["Instances"][0]
    state = instance["State"]["Name"]
    if state in ("stopped", "stopping"):
        return dict(status="already_"+state, utc=now())
    if state != "running" or instance.get("PublicIpAddress") != record["public_ip"]:
        return dict(status="not_stopped_uncertain_instance_identity", utc=now())
    if args.host.split("@")[-1] != instance["PublicIpAddress"]:
        return dict(status="not_stopped_ssh_host_mismatch", utc=now())
    checks = []
    # Require two consecutive quiet samples. Background work is never stopped.
    clean = 0
    for _ in range(6):
        try:
            check = remote_idle(args.host, args.key)
        except (subprocess.TimeoutExpired, OSError):
            check = dict(inspection_complete=False, idle=False, blockers=[dict(kind="inspection_exception")])
        check["utc"] = now(); checks.append(check)
        clean = clean+1 if check.get("inspection_complete") and check.get("idle") else 0
        if clean == 2:
            break
        time.sleep(30)
    if clean != 2:
        return dict(status="not_stopped_remote_not_proven_idle", checks=checks, utc=now())
    if process_state(monitor["process"]) == "running":
        return dict(status="not_stopped_campaign_running", checks=checks, utc=now())
    response = aws(record, "stop-instances", "--instance-ids", instance_id)
    result = dict(status="stop_requested", instance_id=instance_id, response=response, checks=checks, utc=now())
    # AWS acknowledgement is not proof of a completed state transition.
    for _ in range(6):
        time.sleep(10)
        try:
            observed = aws(record, "describe-instances", "--instance-ids", instance_id)
            result["observed_state"] = observed["Reservations"][0]["Instances"][0]["State"]["Name"]
            if result["observed_state"] == "stopped":
                result.update(status="stopped_verified", verified_utc=now())
                break
        except Exception as error:
            result["verification_error"] = type(error).__name__
            break
    return result


def save_progress(args, monitor):
    cells = []
    for folder in sorted(args.campaign.glob("sf_*")):
        if not (folder / "config.json").exists():
            continue
        config = read(folder / "config.json")
        state = read(folder / "state.json") if (folder / "state.json").exists() else {"history":[], "rounds":[]}
        valid = [r["score"] for r in state["history"] if r.get("status") == "valid" and r.get("score") is not None]
        row = dict(run=folder.name, target=config["target"], seed=config["seed"], calls=len(state["history"]),
                   valid_calls=len(valid), best_score=min(valid) if valid else None,
                   complete=(folder / "CELL_DONE").exists(), evaluation=monitor["evaluations"].get(folder.name, {}))
        cells.append(row)
    complete = sum(c["complete"] for c in cells)
    payload = dict(updated_utc=now(), campaign_process=process_state(monitor["process"]),
                   expected_targets=args.expected_targets, completed_targets=complete,
                   status="complete" if complete==args.expected_targets else "running",
                   cells=cells, instance_restoration=monitor.get("instance_restoration"))
    if payload["campaign_process"] != "running" and complete != args.expected_targets:
        payload["status"] = "campaign_terminated_incomplete"
    write(args.campaign / "monitor_progress.json", payload)
    lines = ["# Live Dock-12 campaign progress", "", f"Updated {payload['updated_utc']}. Status: {payload['status']}.",
             f"Completed {complete}/{args.expected_targets} targets. Failures and the 1000 warm queries remain charged.", "",
             "| Target | Calls | Valid | Best raw score | Evaluation |", "|---|---:|---:|---:|---|"]
    for c in cells:
        lines.append(f"| {c['target']} | {c['calls']} | {c['valid_calls']} | {c['best_score']} | {c['evaluation'].get('status','pending')} |")
    for cell in cells:
        if cell["evaluation"].get("status") == "complete":
            result = read(args.campaign / cell["run"] / "evaluation_live/summary.json")
            lines += ["", f"## {cell['target']}: docking scores and greedy portfolios", "",
                      "Lower top-100 docking score is better; higher GPDiv utility is better.", "",
                      "| Domain | Method | Molecular top-100 mean | GPDiv100 |",
                      "|---|---|---:|---:|"]
            for method in result["cells"]:
                for domain, values in method["domains"].items():
                    greedy = next(p for p in values["portfolios"] if p["metric"] == "GPDiv")
                    lines.append(f"| {domain} | {method['method']} | {values['scores']['molecular_top100_mean']} | {greedy.get('value')} |")
        comparisons = cell["evaluation"].get("comparisons", [])
        if comparisons:
            lines += ["", f"## {cell['target']}: strict Morgan Tc < 0.30, OPDiv100 bounds", "",
                      "| Domain | Comparator | SynFactor [lower, upper] | Baseline [lower, upper] | Bound conclusion |",
                      "|---|---|---|---|---|"]
            for r in comparisons:
                lines.append(f"| {r['domain']} | {r['baseline']} | [{r['synfactor_lower']}, {r['synfactor_upper']}] | [{r['baseline_lower']}, {r['baseline_upper']}] | {r['claim']} |")
    lines += ["", "A feasible lower bound is not an exact optimum. Target-wise comparisons retain grid/warm checks.",
              "This monitor never retries docking or restarts the campaign.", ""]
    if monitor.get("instance_restoration"):
        lines.append("Instance restoration: " + monitor["instance_restoration"]["status"])
    (args.campaign / "monitor_progress.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--eval-python", type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--expected-targets", type=int, default=12)
    p.add_argument("--poll-seconds", type=float, default=30)
    p.add_argument("--solver-seconds", type=float, default=30)
    p.add_argument("--restore-instance", action="store_true")
    p.add_argument("--instance-audit", type=Path)
    args = p.parse_args()
    if not 5 <= args.poll_seconds <= 60 or args.solver_seconds <= 0 or args.expected_targets < 1:
        p.error("Need 5–60s polling and positive solver/target counts")
    if args.restore_instance and args.instance_audit is None:
        p.error("Restoration requires --instance-audit")
    args.campaign = args.campaign.resolve(); args.evaluator_root = args.evaluator_root.resolve()
    args.eval_python = args.eval_python.absolute(); args.key = args.key.absolute()
    if args.restore_instance:
        ownership(args.instance_audit)
    state_path = args.campaign / "monitor_state.json"
    evaluator = args.evaluator_root / "experiments/docking/live_evaluate.py"
    binding = dict(campaign=str(args.campaign), pid=args.pid, evaluator_root=str(args.evaluator_root),
                   evaluator_sha256=sha(evaluator), eval_python=str(args.eval_python),
                   solver_seconds=args.solver_seconds, expected_targets=args.expected_targets,
                   host=args.host, restore_instance=args.restore_instance,
                   instance_audit_sha256=sha(args.instance_audit) if args.instance_audit else None)
    with (args.campaign / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            monitor = read(state_path)
            if monitor["binding"] != binding:
                raise ValueError("Supervisor binding changed; inspect prior state before replacing it")
        else:
            monitor = dict(binding=binding, process=bind_process(args.pid,args.campaign),
                           created_utc=now(), evaluations={})
            write(state_path, monitor)
        child, log_handle = None, None
        while True:
            if sha(evaluator) != binding["evaluator_sha256"]:
                raise RuntimeError("Pinned evaluator source changed")
            active = None
            for name, job in monitor["evaluations"].items():
                if job["status"] != "running":
                    continue
                if child is not None and child.pid == job["process"]["pid"]:
                    finished = child.poll() is not None
                else:
                    finished = process_state(job["process"]) != "running"
                if not finished:
                    active = name; continue
                cell = args.campaign/name
                try:
                    config=validate_cell(cell)
                    if sha(cell/'state.json') != job['history_sha256'] or not evaluation_complete(cell/'evaluation_live',job['history_sha256'],config):
                        raise ValueError("Incomplete or mismatched evaluator output")
                    job.update(status="complete", completed_utc=now(),
                               comparisons=portfolio_comparisons(read(cell/'evaluation_live/summary.json')))
                except Exception as error:
                    job.update(status="failed", completed_utc=now(), error=f"{type(error).__name__}: {error}")
                if log_handle is not None:
                    log_handle.close();log_handle=None
                child=None
            pending=[]
            for marker in sorted(args.campaign.glob('sf_*/CELL_DONE')):
                cell=marker.parent
                if cell.name in monitor['evaluations']:
                    continue
                try:
                    config=validate_cell(cell);pending.append((cell,config))
                except Exception as error:
                    monitor['evaluations'][cell.name]=dict(status='failed',error=f'{type(error).__name__}: {error}')
            if active is None and pending:
                cell,config=pending[0];out=cell/'evaluation_live';out.mkdir(exist_ok=True)
                log_handle=(out/'monitor_evaluation.log').open('a')
                command=[str(args.eval_python),'-m','experiments.docking.live_evaluate','--history',str(cell/'state.json'),
                         '--target',config['target'],'--seed',str(config['seed']),'--out',str(out),
                         '--grid-sha256',config['grid_sha256'],'--warm-sha256',config['warm']['csv_sha256'],
                         '--max-seconds',str(args.solver_seconds)]
                child=subprocess.Popen(command,cwd=args.evaluator_root,stdout=log_handle,stderr=subprocess.STDOUT,
                                       env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                identity=process_identity(child.pid)
                if identity is None:
                    monitor['evaluations'][cell.name]=dict(status='failed',error='Evaluator exited during launch')
                    log_handle.close();log_handle=None;child=None
                else:
                    monitor['evaluations'][cell.name]=dict(status='running',started_utc=now(),process=identity,
                        history_sha256=sha(cell/'state.json'),output=str(out));active=cell.name
            write(state_path,monitor);save_progress(args,monitor)
            if process_state(monitor['process'])!='running' and active is None and not pending:
                if args.restore_instance and 'instance_restoration' not in monitor:
                    try:monitor['instance_restoration']=restore_instance(args,monitor)
                    except Exception as error:monitor['instance_restoration']=dict(status='not_stopped_inspection_error',error=f'{type(error).__name__}: {error}',utc=now())
                    write(state_path,monitor);save_progress(args,monitor)
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
