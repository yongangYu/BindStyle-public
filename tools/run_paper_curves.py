"""Plan/run isolated GPU experiments; no CUDA imports in the sweep process."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.paper_curve_protocol import PROTOCOL, digest, make_jobs


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def collect(output, plan):
    rows = []
    for job in plan["jobs"]:
        folder = output / job["id"]
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "planned"}
        measurements = folder / "measurements.jsonl"
        records = []
        if measurements.exists():
            for line in measurements.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if status["status"] == "complete":
                        raise
        if not records:
            records = [{"metric": "", "value": None, "status": status["status"]}]
        for row in records:
            rows.append({**{k: v for k, v in job.items() if k != "windows"}, **row,
                         "job_status": status["status"], "error": status.get("error", "")})
    fields = sorted({k for row in rows for k in row})
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="config/3dgs.yaml,config/dnerf/jumpingjacks.yaml,config/style/split_vgg_geometry.yaml,config/evaluation/paper.yaml,config/evaluation/curves.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--execute", action="store_true", help="Default only writes the plan")
    parser.add_argument("--resume", action="store_true", help="Skip completed jobs; restart failed jobs from the original model")
    parser.add_argument("--smoke", action="store_true", help="Small engineering check, never a paper result")
    args = parser.parse_args()
    from omegaconf import OmegaConf
    os.chdir(ROOT)
    cfg = OmegaConf.merge(*[OmegaConf.load(path) for path in args.configs.split(",")], OmegaConf.from_dotlist(args.set))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    curves = cfg["curves"]
    if curves.get("protocol") != PROTOCOL:
        raise ValueError("Unknown curve protocol")
    if args.smoke:
        curves.update(sh_updates=4, geometry_updates=3, schedule_warmup=1, probe_frames=2)
        curves["A"].update(windows=[2], resolutions=[256], warmup_updates=1, measured_updates=2)
        curves["C"]["coverage"] = [1]
    seeds = args.seeds or ([1234] if args.smoke else curves["seeds"])
    jobs = make_jobs(curves, list(dict.fromkeys(args.groups)), seeds)
    plan = {"protocol": PROTOCOL, "smoke": args.smoke, "config": cfg, "jobs": jobs}
    plan["signature"] = digest(plan)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text())["signature"] != plan["signature"]:
            raise ValueError("Output directory contains a different experiment plan. Use a new output.")
        if args.execute and not args.resume and any((output / j["id"] / "status.json").exists() for j in jobs):
            raise ValueError("Existing runs found; use --resume or a new output directory")
    save_json(plan_path, plan)
    print(f"{len(jobs)} jobs; output={output}; smoke={args.smoke}", flush=True)
    if not args.execute:
        print("Plan only. Add --execute to run; no GPU/model initialization performed.")
        return
    failures = 0
    for job in jobs:
        folder = output / job["id"]
        folder.mkdir(exist_ok=True)
        status_path = folder / "status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous["status"] == "complete" and args.resume:
                continue
            # Preserve partial measurements on restart instead of mixing trials.
            for filename in ("measurements.jsonl", "status.json", "run.log", "updates.jsonl",
                             "manifest.json", "effective_config.json", "checkpoints", "paper_eval"):
                source = folder / filename
                if source.exists():
                    from datetime import datetime, timezone
                    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                    source.replace(folder / f"{filename}.{suffix}.previous")
        save_json(folder / "job.json", {"job": job, "config": cfg, "smoke": args.smoke, "plan_signature": plan["signature"]})
        print(f"Running {job['id']} (log: {folder / 'run.log'})", flush=True)
        try:
            with (folder / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-u", str(ROOT / "tools/train_paper_curve.py"), str(folder)],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            save_json(status_path, {"status": "interrupted", "error": "Interrupted by user"})
            collect(output, plan)
            raise
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") in {None, "running"}:
            save_json(status_path, {"status": "failed", "error": f"Worker exit code {process.returncode}; inspect run.log"})
            status = json.loads(status_path.read_text())
        if status["status"] != "complete":
            failures += 1
        print(f"  {status['status']}", flush=True)
        collect(output, plan)
    collect(output, plan)
    print(f"Finished sweep; {failures} unsuccessful jobs. Measurements: {output / 'results.csv'}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
