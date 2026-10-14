"""拓扑 1 最小 demo 的自检用例。"""
from pathlib import Path

import pytest

from agentcc import Topology, plan_all_gather, plan_all_reduce, replay

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / "topic3/拓扑与输出规范/拓扑1/拓扑1输入.json"


@pytest.fixture(scope="module")
def topo() -> Topology:
    return Topology.load(TOPO)


@pytest.mark.parametrize(
    "operator,split",
    [
        ("all-gather", 1),
        ("all-gather", 4),
        ("all-reduce", 1),
        ("all-reduce", 4),
    ],
)
def test_solution_replay(topo: Topology, operator: str, split: int) -> None:
    if operator == "all-gather":
        plan = plan_all_gather(topo, split=split)
    else:
        plan = plan_all_reduce(topo, split=split)
    report = replay(topo, plan, split=split)
    assert report.ok, "\n".join(report.errors)
    assert report.total_time_us > 0
    assert len(report.round_times_us) == plan["round_count"]


def test_replay_rejects_broken_schedule(topo: Topology) -> None:
    plan = plan_all_gather(topo, split=1)
    # 伪造：让 r0 发送一个它并不持有的分片
    plan["rounds"][0]["transfers"][0]["chunk_ranges"] = ["r127_c0"]
    report = replay(topo, plan, split=1)
    assert not report.ok
