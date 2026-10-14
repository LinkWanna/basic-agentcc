#!/usr/bin/env python3
"""AgentCC 最小求解 demo（默认输入：拓扑 1 / CloudMatrix）。

用法：
    uv run solve.py                                  # AllGather（默认）
    uv run solve.py --operator all-reduce --split 4
    uv run solve.py --check out/topology1_all_gather_solution.json

流程：读拓扑输入 → 构造调度 → 本地 Replay 校验与计时 → 输出 solution JSON。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentcc import (
    Topology,
    parse_data_size,
    plan_all_gather,
    plan_all_reduce,
    replay,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = ROOT / "topic/拓扑与输出规范/拓扑1/拓扑1输入.json"


def build_plan(topo: Topology, operator: str, split: int, fanout: int) -> dict:
    if operator == "all-gather":
        return plan_all_gather(topo, split=split, fanout=fanout)
    return plan_all_reduce(topo, split=split, fanout=fanout)


def attach_metrics(plan: dict, report) -> None:
    plan["metrics"] = {
        "max_round": len(plan["rounds"]),
        "round_times": [t / 1e6 for t in report.round_times_us],
        "total_time": report.total_time_us / 1e6,
    }


def print_header(topo: Topology, operator: str, split: int, fanout: int) -> None:
    print(
        f"拓扑: {topo.raw.get('description', '?')} | "
        f"ranks={len(topo.ranks)} switches={len(topo.switches)} "
        f"edges={len(topo.raw.get('edges', []))}"
    )
    print(
        f"算子: {operator} | split={split} | fanout={fanout} | "
        f"chunk={topo.per_rank_bytes / split / 1024:.1f} KB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentCC 最小求解 demo")
    parser.add_argument(
        "--topology", default=str(DEFAULT_TOPOLOGY), help="拓扑输入 JSON"
    )
    parser.add_argument(
        "--operator", choices=["all-gather", "all-reduce"], default="all-gather"
    )
    parser.add_argument(
        "--split", type=int, default=None, help="切分数（默认取候选最优）"
    )
    parser.add_argument(
        "--fanout", type=int, default=7, help="每条逻辑发送的并行路径数"
    )
    parser.add_argument("--data-size", default=None, help="覆盖每 rank 数据量，如 1MB")
    parser.add_argument("--out", default=None, help="输出 solution JSON 路径")
    parser.add_argument("--check", default=None, help="只校验已有 solution JSON")
    args = parser.parse_args(argv)

    topo = Topology.load(args.topology)
    if args.data_size:
        topo.per_rank_bytes = parse_data_size(args.data_size)

    if args.check:
        solution = json.loads(Path(args.check).read_text(encoding="utf-8"))
        report = replay(topo, solution)
        print(f"校验 {args.check}")
        print(report.summary())
        return 0 if report.ok else 1

    operator = args.operator
    if args.split is not None:
        split = args.split
    elif operator == "all-gather":
        split = min(topo.split_candidates)
    else:
        split = max(topo.split_candidates)

    print_header(topo, operator, split, args.fanout)

    plan = build_plan(topo, operator, split, args.fanout)
    report = replay(topo, plan, split=split)
    attach_metrics(plan, report)

    print(f"轮数: {plan['round_count']} | transfers: {report.transfer_count}")
    print("本地 Replay: " + report.summary())

    if not report.ok:
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "out" / (f"topology1_{operator.replace('-', '_')}_solution.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
