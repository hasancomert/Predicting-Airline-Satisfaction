"""Bundle src/ into a private Kaggle kernel, run commands on a Kaggle GPU, fetch the OOF/test files.

Usage:
  python kaggle/gpu_kernel.py push <slug> "<command>" ["<command>" ...]   # build + push
  python kaggle/gpu_kernel.py pushcpu <slug> "<command>" ...   # same on a CPU session (no GPU quota;
                                                               # 4 cores, ~30 GB RAM)
  python kaggle/gpu_kernel.py status <slug>
  python kaggle/gpu_kernel.py fetch <slug>          # download outputs into oof/ and preds/
Commands run from the project root inside the kernel, e.g.
  "python src/train.py xgb base,te1 orig=1 preset=t1 lr=0.03 folds=10 device=cuda"
The kernel attaches the competition data and arseniyshutko/binary-aviation-satisfaction-129k, keeps
the project and the TE computation in /tmp (TE_CACHE=0), and writes only *.npy and logs to output.
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "hasancmert"
BUILD = ROOT / "kaggle" / "build"

RUNNER = r'''
import base64, glob, os, shutil, subprocess, sys, time
FILES = __FILES__
COMMANDS = __COMMANDS__
proj = "/tmp/proj"
os.makedirs(proj + "/src", exist_ok=True)
for name, b64 in FILES.items():
    open(f"{proj}/src/{name}", "wb").write(base64.b64decode(b64))
os.makedirs(proj + "/data/orig", exist_ok=True)
def find(name, hint):
    hits = sorted(p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p)
    assert hits, (name, hint)
    return hits[0]
for f in ("train.csv", "test.csv", "sample_submission.csv"):
    os.symlink(find(f, "playground-series-s6e10"), f"{proj}/data/{f}")
os.symlink(find("data.csv", "aviation"), f"{proj}/data/orig/data.csv")
if any("realmlp" in c for c in COMMANDS):
    subprocess.run("pip install -q pytabkit", shell=True)
env = dict(os.environ, TE_CACHE="0")
out = "/kaggle/working"
for c in COMMANDS:
    t0 = time.time()
    print(f"### {c}", flush=True)
    r = subprocess.run(c, shell=True, cwd=proj, env=env)
    print(f"### exit {r.returncode} after {time.time() - t0:.0f}s", flush=True)
    for d in ("oof", "preds"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
        for p in glob.glob(f"{proj}/{d}/*.npy"):
            shutil.copy(p, f"{out}/{d}/")
    if os.path.exists(f"{proj}/experiments.md"):
        shutil.copy(f"{proj}/experiments.md", f"{out}/experiments_rows.md")
'''


def push(slug, commands, gpu=True):
    d = BUILD / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    files = {p.name: base64.b64encode(p.read_bytes()).decode() for p in (ROOT / "src").glob("*.py")}
    code = RUNNER.replace("__FILES__", repr(files)).replace("__COMMANDS__", repr(commands))
    (d / "run.py").write_text(code)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu,
            "enable_internet": True, "competition_sources": ["playground-series-s6e10"],
            "dataset_sources": ["arseniyshutko/binary-aviation-satisfaction-129k"],
            "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


def fetch(slug):
    d = BUILD / slug / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(d)], check=True)
    for sub in ("oof", "preds"):
        for p in list(d.glob(f"{sub}/*.npy")) + list(d.glob(f"**/{sub}/*.npy")):
            shutil.copy(p, ROOT / sub / p.name)
            print("fetched", sub, p.name)


if __name__ == "__main__":
    cmd, slug = sys.argv[1], sys.argv[2]
    if cmd == "push":
        push(slug, sys.argv[3:])
    elif cmd == "pushcpu":
        push(slug, sys.argv[3:], gpu=False)
    elif cmd == "status":
        subprocess.run(["kaggle", "kernels", "status", f"{USER}/{slug}"])
    elif cmd == "fetch":
        fetch(slug)
