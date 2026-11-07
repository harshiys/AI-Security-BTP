#!/usr/bin/env python3
"""
Run a SMALL, configurable subset of TAMAS scenarios and save the REAL execution
logs, using Groq's OpenAI-compatible API (model default: openai/gpt-oss-20b).

    python tamas_run_subset.py --tamas_dir TAMAS --attack DPI --scenario news --config RoundRobin --n 10
    python tamas_run_subset.py ... --dry_run      # print exact commands, run nothing
    python tamas_run_subset.py ... --overwrite    # re-run even if a log already exists

Logs are written to  <out>/logs/<attack>/<config>/<attack>_<scenario>_<i>.txt
which is exactly what mla_pipeline.py reads. A status JSON is written to
<out>/real_execution_status.json (model/provider, per-run exit code, log path,
byte count, overall status).

This runner calls TAMAS's own autogen_run.py. Use the SentinelMesh Groq patch
(patches/autogen_run.py copied over TAMAS/autogen_run.py) so the client targets
Groq and tools are wired for all five scenarios.

INTEGRITY: never fabricates a log or a label. If anything is missing it stops,
writes the blocker to the status JSON, and exits 2.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

ATTACKS = ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]
SCENARIOS = ["education", "finance", "healthcare", "legal", "news"]
DEFAULT_MODEL = "openai/gpt-oss-20b"
PROVIDER = "groq"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def preflight(a):
    problems = []
    if not os.path.exists(os.path.join(a.tamas_dir, "autogen_run.py")):
        problems.append(
            f"{a.tamas_dir}/autogen_run.py not found "
            "(git clone --depth 1 https://github.com/microsoft/TAMAS.git TAMAS, "
            "then copy patches/autogen_run.py over it)")
    try:
        import autogen_agentchat, autogen_ext  # noqa: F401
    except Exception as e:
        problems.append(
            f"autogen not importable ({type(e).__name__}): "
            'pip install autogen-agentchat "autogen-ext[openai]"')
    if not os.getenv("GROQ_API_KEY"):
        problems.append(
            "GROQ_API_KEY is not set. PowerShell: $env:GROQ_API_KEY='gsk_...' | "
            "Git Bash: export GROQ_API_KEY=gsk_...")
    task_file = os.path.join(a.tamas_dir, "data", a.attack, f"{a.scenario}_{a.attack}.json")
    if not os.path.exists(task_file):
        problems.append(f"task file not found: {task_file}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamas_dir", default="TAMAS")
    ap.add_argument("--attack", default="DPI", choices=ATTACKS)
    ap.add_argument("--scenario", default="news", choices=SCENARIOS)
    ap.add_argument("--config", default="RoundRobin",
                    choices=["RoundRobin", "Swarm", "Magentic_one"])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Groq model id (default openai/gpt-oss-20b)")
    ap.add_argument("--n", type=int, default=10,
                    help="how many task instances (first n) to run")
    ap.add_argument("--out", default="tamas_real_groq")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per instance")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run and overwrite logs that already exist")
    a = ap.parse_known_args(argv)[0]

    task_file = os.path.join(a.tamas_dir, "data", a.attack, f"{a.scenario}_{a.attack}.json")
    log_dir = os.path.join(a.out, "logs", a.attack, a.config)
    tools_dir = os.path.abspath(os.path.join(a.tamas_dir, "data", "tools", "autogen"))

    tasks = None
    if os.path.exists(task_file):
        tasks = json.load(open(task_file, encoding="utf-8"))[: a.n]
    n_plan = len(tasks) if tasks is not None else a.n
    print(f"Plan: {n_plan} run(s) of {a.attack}/{a.scenario}/{a.config} "
          f"| model={a.model} provider={PROVIDER} -> {log_dir}")
    for i in range(n_plan):
        log = os.path.join(log_dir, f"{a.attack}_{a.scenario}_{i}.txt")
        print(f"  (cd {a.tamas_dir} && PYTHONPATH={tools_dir} "
              f"{sys.executable} autogen_run.py --json_data <task_{i}.json> "
              f"--model {a.model} --config {a.config} > {log} 2>&1)")
    if a.dry_run:
        return 0

    os.makedirs(a.out, exist_ok=True)
    status_path = os.path.join(a.out, "real_execution_status.json")
    problems = preflight(a)
    if problems:
        json.dump(dict(status="blocked", provider=PROVIDER, model=a.model,
                       blockers=problems, when_utc=_now(),
                       command_to_continue=(
                           f"python tamas_run_subset.py --tamas_dir {a.tamas_dir} "
                           f"--attack {a.attack} --scenario {a.scenario} "
                           f"--config {a.config} --model {a.model} --n {a.n} "
                           f"--out {a.out}")),
                  open(status_path, "w", encoding="utf-8"), indent=2)
        print("\nBLOCKED. No logs were produced and none were invented:")
        for p in problems:
            print("  -", p)
        print(f"Details: {status_path}")
        return 2

    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ,
               PYTHONPATH=tools_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))

    done = []
    for i, task in enumerate(tasks):
        log_path = os.path.join(log_dir, f"{a.attack}_{a.scenario}_{i}.txt")
        if os.path.exists(log_path) and not a.overwrite:
            print(f"  run {i}: SKIP (log exists; use --overwrite to replace)")
            done.append(dict(index=i, log=log_path, returncode="skipped",
                             bytes=os.path.getsize(log_path)))
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as tf:
            json.dump(task, tf)
            tf_path = tf.name
        cmd = [sys.executable, "autogen_run.py", "--json_data", tf_path,
               "--model", a.model, "--config", a.config]
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                rc = subprocess.run(cmd, cwd=a.tamas_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            except Exception as e:
                log.write(f"\n[runner error] {type(e).__name__}: {e}\n")
                rc = f"error:{type(e).__name__}"
        os.remove(tf_path)
        size = os.path.getsize(log_path)
        done.append(dict(index=i, log=log_path, returncode=rc, bytes=size))
        print(f"  run {i}: rc={rc}, {size} bytes")

    ok = [d for d in done if d["returncode"] in (0, "skipped") and d["bytes"] > 0]
    overall = "completed" if len(ok) == len(done) else "partial"
    json.dump(dict(status=overall, provider=PROVIDER, model=a.model,
                   attack=a.attack, scenario=a.scenario, config=a.config,
                   runs=done, n_ok=len(ok), n_total=len(done), when_utc=_now()),
              open(status_path, "w", encoding="utf-8"), indent=2)
    print(f"\nStatus: {overall} ({len(ok)}/{len(done)} ok) -> {status_path}")
    print(f"Next: python mla_pipeline.py --tamas_dir {a.tamas_dir} "
          f"--logs_dir {a.out}/logs --out out_tamas_groq")
    return 0


if __name__ == "__main__":
    sys.exit(main())
