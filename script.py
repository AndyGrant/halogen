#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

HALOGEN_DIR  = Path(__file__).resolve().parent
NETWORKS_DIR = HALOGEN_DIR.parent / "networks"
SRC_DIR      = HALOGEN_DIR / "src"
ARCH_HPP     = SRC_DIR / "network" / "arch.hpp"
VERBATIM_CPP = SRC_DIR / "tools" / "verbatim.cpp"
LOGS_DIR     = HALOGEN_DIR / "shuffle_logs"
BENCH_DIR    = HALOGEN_DIR / "bench_logs"
REMOTE_HOST = "gpu303"

def run(cmd, verbose=False, **kwargs):
    if verbose:
        print(f"$ {cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd)}")
    else:
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"command failed (exit {result.returncode})")
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp",      type=str, required=True, help="Experiment Name (exp14) ")
    parser.add_argument("--stage",    type=int, default=2,     help="Training Stage (Default: 2")
    parser.add_argument("--sb",       type=int, default=100,   help="SuperBatch Number (Default: 100)")
    parser.add_argument("--ftsize",   type=int, default=768,   help="Overwride src/network/arch.hpp's FT_SIZE (Default: 768)")

    parser.add_argument("--override",      action="store_true", help="Force a rebuild and reshuffle")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print commands and subprocess output")
    parser.add_argument("-y", "--yes",     action="store_true", help="Actually run the commit command")
    args = parser.parse_args()

    # -- Step 0 -- : Setup paths, resolved relative to this script, not the cwd.
    network      = f"{args.exp}-stage{args.stage}-{args.sb}.bb"
    network_path = NETWORKS_DIR / network
    log_path     = LOGS_DIR / f"{network}.logs"
    bench_path   = BENCH_DIR / f"{network}.logs"

    # -- Step 1 -- : Enforce --override by removing the existing log, forcing a rebuild and reshuffle.
    if args.override and log_path.exists():
        print(f"--override: removing {log_path}")
        log_path.unlink()

    # -- Step 2 -- : Download the network file if we don't already have it.
    if network_path.exists():
        print(f"network already present: {network_path}")
    else:
        NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
        remote = (
            f"{REMOTE_HOST}:training/virifiles/bullet-{args.exp}/"
            f"bullet-{args.exp}-stage{args.stage}-{args.sb}/quantised.bin"
        )
        run(["scp", remote, str(network_path)], verbose=args.verbose)

    # -- Step 3 -- : Update FT_SIZE in arch.hpp before building.
    text = ARCH_HPP.read_text()
    text, count = re.subn(
        r"(constexpr size_t FT_SIZE\s*=\s*)\d+(\s*;)",
        rf"\g<1>{args.ftsize}\g<2>",
        text,
    )
    if count != 1:
        sys.exit(f"expected exactly one FT_SIZE definition in {ARCH_HPP}, found {count}")
    ARCH_HPP.write_text(text)
    print(f"set FT_SIZE = {args.ftsize} in {ARCH_HPP}")

    # -- Step 4 -- : Execute shuffle_network, building the engine against the network.
    if not log_path.exists():
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        make_cmd = (
            f"make -j shuffle EXE=halogen EVALFILE={network_path} "
            f"&& ./halogen shuffle_network > {log_path}"
        )
        run(make_cmd, shell=True, cwd=SRC_DIR, executable="/bin/bash", verbose=args.verbose)

    # -- Step 5 -- : Apply the permutation. It is the second to last line of the log.
    perm_line = log_path.read_text().splitlines()[-2]
    values = [int(v) for v in re.findall(r"\d+", perm_line)]
    if len(values) != args.ftsize // 2:
        sys.exit(f"expected {args.ftsize // 2} permutation values, found {len(values)}")

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

    # -- Step 6 -- : Rebuild with the permuted network and bench it.
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    bench_cmd = (
        f"make -j release EXE=halogen EVALFILE={network_path} "
        f"&& ./halogen bench > {bench_path}"
    )
    run(bench_cmd, shell=True, cwd=SRC_DIR, executable="/bin/bash", verbose=args.verbose)
    print(f"wrote bench results to {bench_path}")

    # -- Step 7 -- : Commit. Bench is the first value of the last line of the bench log.
    bench = bench_path.read_text().splitlines()[-1].split()[0]
    commit_cmd = f'git checkout {network} && git add src/* && git commit -m "{network} Bench {bench}"'

    if not args.yes:
        print("\nTo commit:")
        print(commit_cmd)
    else:
        run(commit_cmd, shell=True, cwd=HALOGEN_DIR, executable="/bin/bash", verbose=args.verbose)
        log = subprocess.run(
            ["git", "log", "-1"], cwd=HALOGEN_DIR, capture_output=True, text=True
        )
        print(log.stdout)
