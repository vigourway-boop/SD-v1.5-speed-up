"""Check the environment then start the existing paired generation entry point."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from project_health import check_project, print_health


ROOT = Path(__file__).resolve().parent
FAST_PROFILE = ROOT / "profiles" / "temporal_fast_100.json"


def select_mode():
    print("\n请选择加速模式 / Select acceleration mode")
    print("  1. 默认加速 / Default")
    print("  2. 快速策略 / Fast")
    while True:
        choice = input("请输入 1 或 2，直接回车选择 1: ").strip() or "1"
        if choice in {"1", "2"}:
            return choice
        print("输入无效，请输入 1 或 2。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--decision-backend", choices=("auto", "pc", "pynq"),
                        default=os.environ.get("SD_DECISION_BACKEND", "auto"))
    parser.add_argument("--pynq-host", default=os.environ.get("PYNQ_HOST", "192.168.2.99"))
    parser.add_argument("--pynq-port", type=int, default=int(os.environ.get("PYNQ_PORT", "9000")))
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument("--controller-profile", type=Path)
    strategy.add_argument("--mode", choices=("1", "2"),
                          help="1: default acceleration; 2: fast profile; skips the menu")
    args, remaining = parser.parse_known_args()
    os.chdir(ROOT)
    if args.mode is None and args.controller_profile is None and not args.check_only:
        try:
            args.mode = select_mode()
        except EOFError:
            print("\n未收到模式选择。可使用 --mode 1 或 --mode 2 直接启动。", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\n已取消运行。")
            return 130
    if args.mode == "2":
        args.controller_profile = FAST_PROFILE
        print("已选择模式 2：快速策略 / Fast")
    elif args.mode == "1":
        print("已选择模式 1：默认加速 / Default")
    result = check_project(args.decision_backend, args.pynq_host, args.pynq_port, args.controller_profile)
    print_health(result)
    if result["errors"]:
        return 1
    if args.check_only:
        return 0
    command = [sys.executable, str(ROOT / "combined_speed_test.py"), "--with-baseline",
               "--decision-backend", args.decision_backend,
               "--pynq-host", args.pynq_host, "--pynq-port", str(args.pynq_port)]
    if args.controller_profile:
        command += ["--controller-profile", str(args.controller_profile)]
    command += remaining
    sys.stdout.flush()
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
