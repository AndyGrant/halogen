#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

HALOGEN_DIR  = Path(__file__).resolve().parent
NETWORKS_DIR = HALOGEN_DIR.parent / "networks"
SRC_DIR      = HALOGEN_DIR / "src"
LOGS_DIR     = HALOGEN_DIR / "shuffle_logs"
BENCH_DIR    = HALOGEN_DIR / "bench_logs"

ARCH_HPP     = SRC_DIR / "network" / "arch.hpp"
VERBATIM_CPP = SRC_DIR / "tools" / "verbatim.cpp"

REMOTE_HOST = "gpu303"
ENGINE      = "Halogen"

def url_join(*parts):
    return "/".join(str(p).rstrip("/") for p in parts) + "/"

def run(cmd, verbose=False, **kwargs):
    if verbose:
        print(f"$ {cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd)}")
    else:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"command failed (exit {result.returncode})\ncommand was: {cmd}")
    return result

def download_network(args, network_path):
    if network_path.exists():
        print(f"network already present: {network_path}")
        return
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    remote = (
        f"{REMOTE_HOST}:training/virifiles/bullet-{args.exp}/"
        f"bullet-{args.exp}-stage{args.stage}-{args.sb}/quantised.bin"
    )
    run(["scp", remote, str(network_path)], verbose=args.verbose)

def upload_to_torchbench(args, network, network_path):
    if not (args.username and args.password and args.server):
        sys.exit("set TORCHBENCH_USERNAME, TORCHBENCH_PASSWORD, and TORCHBENCH_SERVER (or pass -U/-P/-S)")
    creds = {"username": args.username, "password": args.password}
    nets = requests.post(url_join(args.server, "api", "networks", ENGINE), data=creds).json()["networks"]
    if any(n.get("name") == network for n in nets):
        print(f"network already on TorchBench: {network}")
        return
    if not args.yes and input(f"Upload {network} to TorchBench? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        return
    data = {**creds, "engine": ENGINE, "name": network, "action": "UPLOAD_NETWORK"}
    with open(network_path, "rb") as netfile:
        r = requests.post(url_join(args.server, "scripts"), data=data, files={"netfile": netfile})
    print(f"upload status code: {r.status_code}")
    for label, css in (("Error", "error-message"), ("Status", "status-message")):
        pattern = rf'<div class="{css}">\s*<pre>(.*?)</pre>\s*</div>'
        if m := re.findall(pattern, r.text, re.DOTALL):
            print(f"{label}: {m[0].strip()}")

def set_ft_size(ftsize):
    text = ARCH_HPP.read_text()
    text, count = re.subn(
        r"(constexpr size_t FT_SIZE\s*=\s*)\d+(\s*;)",
        rf"\g<1>{ftsize}\g<2>",
        text,
    )
    if count != 1:
        sys.exit(f"expected exactly one FT_SIZE definition in {ARCH_HPP}, found {count}")
    ARCH_HPP.write_text(text)
    print(f"set FT_SIZE = {ftsize} in {ARCH_HPP}")

def shuffle_network(args, network_path, log_path):
    if log_path.exists():
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    make_cmd = (
        f"make -j shuffle EXE=halogen EVALFILE={network_path} "
        f"&& ./halogen shuffle_network > {log_path}"
    )
    run(make_cmd, shell=True, cwd=SRC_DIR, executable="/bin/bash", verbose=args.verbose)

def apply_permutation(log_path, ftsize):
    perm_line = log_path.read_text().splitlines()[-2]
    values = [int(v) for v in re.findall(r"\d+", perm_line)]
    if len(values) != ftsize // 2:
        sys.exit(f"expected {ftsize // 2} permutation values, found {len(values)}")

    indent = " " * 8
    rows = [values[i:i + 16] for i in range(0, len(values), 16)]
    body = "\n".join(indent + ", ".join(f"{v:4d}" for v in row) + "," for row in rows)

    text = VERBATIM_CPP.read_text()
    text, count = re.subn(
        r"(shuffle_order = \{\n).*?(\n[ \t]*\};)",
        rf"\g<1>{body}\g<2>",
        text,
        flags=re.DOTALL,
    )
    if count != 1:
        sys.exit(f"expected exactly one shuffle_order array in {VERBATIM_CPP}, found {count}")
    VERBATIM_CPP.write_text(text)
    print(f"updated shuffle_order ({len(values)} values) in {VERBATIM_CPP}")

def bench_network(args, network_path, bench_path):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    bench_cmd = (
        f"make -j release EXE=halogen EVALFILE={network_path} "
        f"&& ./halogen bench > {bench_path}"
    )
    run(bench_cmd, shell=True, cwd=SRC_DIR, executable="/bin/bash", verbose=args.verbose)
    print(f"wrote bench results to {bench_path}")

def commit(args, network, bench_path):
    bench = bench_path.read_text().splitlines()[-1].split()[0]
    commit_cmd = (
        f'(git checkout {network} 2>/dev/null || git checkout -b {network}) '
        f'&& git add src/* && git commit -m "{network} Bench {bench}"'
    )
    print("\nTo commit:")
    print(commit_cmd)
    if not args.yes and input("Run it? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        return
    run(commit_cmd, shell=True, cwd=HALOGEN_DIR, executable="/bin/bash", verbose=args.verbose)
    log = subprocess.run(["git", "log", "-1"], cwd=HALOGEN_DIR, capture_output=True, text=True)
    print(log.stdout)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--exp",      type=str, required=True, help="Experiment Name (exp14)")
    experiment.add_argument("--stage",    type=int, default=2,     help="Training Stage (Default: 2)")
    experiment.add_argument("--sb",       type=int, default=100,   help="SuperBatch Number (Default: 100)")
    experiment.add_argument("--ftsize",   type=int, default=768,   help="Override src/network/arch.hpp's FT_SIZE (Default: 768)")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--override",      action="store_true", help="Force a rebuild and reshuffle")
    behaviour.add_argument("-v", "--verbose", action="store_true", help="Print commands and subprocess output")
    behaviour.add_argument("-y", "--yes",     action="store_true", help="Assume Y to prompts (skip commit confirmation)")

    torchbench = parser.add_argument_group("torchbench")
    torchbench.add_argument("--username", default=os.environ.get("TORCHBENCH_USERNAME"), help="TorchBench username (or TORCHBENCH_USERNAME)")
    torchbench.add_argument("--password", default=os.environ.get("TORCHBENCH_PASSWORD"), help="TorchBench password (or TORCHBENCH_PASSWORD)")
    torchbench.add_argument("--server",   default=os.environ.get("TORCHBENCH_SERVER"),   help="TorchBench server   (or TORCHBENCH_SERVER  )")
    args = parser.parse_args()

    # -- Step 0 -- : Setup paths, resolved relative to this script, not the cwd.
    network      = f"{args.exp}-stage{args.stage}-{args.sb}.nn"
    network_path = NETWORKS_DIR / network
    log_path     = LOGS_DIR / f"{network}.logs"
    bench_path   = BENCH_DIR / f"{network}.logs"

    # -- Step 1 -- : Enforce --override by removing the existing log, forcing a rebuild and reshuffle.
    if args.override and log_path.exists():
        print(f"--override: removing {log_path}")
        log_path.unlink()

    download_network(args, network_path)                 # -- Step 2 --
    upload_to_torchbench(args, network, network_path)    # -- Step 3 --
    set_ft_size(args.ftsize)                             # -- Step 4 --
    shuffle_network(args, network_path, log_path)        # -- Step 5 --
    apply_permutation(log_path, args.ftsize)             # -- Step 6 --
    bench_network(args, network_path, bench_path)        # -- Step 7 --
    commit(args, network, bench_path)                    # -- Step 8 --
