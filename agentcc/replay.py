"""本地 Replay：按赛题文档的口径校验方案正确性并计算通信耗时。

校验内容：
- chunks / initial_possession / final_chunks 与任务语义一致；
- 每个 transfer 的持有性、接收唯一性、发送唯一性、轮内状态只读；
- route_id 路径端点、交换机中继、无重复节点、边必须来自拓扑输入；
- AllReduce 规约贡献者集合不重复、最终覆盖全集。

计时口径（微秒）：
    传输耗时 = startup + 交换机跳数 × switch_overhead + 10^6 × Σ(字节数 / bw)
    本轮耗时 = max(max_t(传输耗时_t), A + max_e(边忙碌时间_e))
    A = max_t(startup + h_t × switch_overhead)
    total_time = Σ 本轮耗时
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .topology import Topology

_TERM_RE = re.compile(r"^r(\d+)_c(\d+)$")
_SUM_RE = re.compile(r"^c(\d+)_sum$")


@dataclass
class ReplayReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    round_times_us: list[float] = field(default_factory=list)
    total_time_us: float = 0.0
    transfer_count: int = 0

    @property
    def total_time_s(self) -> float:
        return self.total_time_us / 1e6

    def summary(self) -> str:
        lines = ["PASS" if self.ok else "FAIL"]
        lines.append(
            f"transfers: {self.transfer_count}, rounds: {len(self.round_times_us)}"
        )
        if self.round_times_us:
            lines.append(
                "round_times(us): " + ", ".join(f"{t:.2f}" for t in self.round_times_us)
            )
        lines.append(
            f"total_time: {self.total_time_us:.2f} us ({self.total_time_s * 1e3:.4f} ms)"
        )
        for err in self.errors[:20]:
            lines.append(f"  - {err}")
        if len(self.errors) > 20:
            lines.append(f"  ... 共 {len(self.errors)} 条错误")
        return "\n".join(lines)


def _parse_expr(expr: str) -> tuple[int, frozenset[int]] | None:
    """解析聚合分片表达式，返回 (slot, 贡献者集合)。"""
    m = _SUM_RE.match(expr.strip())
    if m:
        return int(m.group(1)), frozenset()  # 全集在调用处按参与 rank 展开
    slot = None
    contributors: set[int] = set()
    for term in expr.split("+"):
        m = _TERM_RE.match(term.strip())
        if not m:
            return None
        rank, s = int(m.group(1)), int(m.group(2))
        if slot is None:
            slot = s
        elif s != slot:
            return None
        contributors.add(rank)
    if slot is None:
        return None
    return slot, frozenset(contributors)


def _segment_nodes(text: str, names: list[str]) -> list[str] | None:
    """把 "L1-N0-S0-L2-P0-S0" 之类的中段文本切回节点名序列。"""
    memo: dict[int, list[str] | None] = {}

    def rec(pos: int) -> list[str] | None:
        if pos == len(text):
            return []
        if pos in memo:
            return memo[pos]
        for name in names:
            if not text.startswith(name, pos):
                continue
            end = pos + len(name)
            if end < len(text) and text[end] != "-":
                continue
            sub = rec(end + 1) if end < len(text) else []
            if sub is not None:
                memo[pos] = [name] + sub
                return memo[pos]
        memo[pos] = None
        return None

    return rec(0)


def _resolve_route(
    topo: Topology,
    route_id: str,
    round_no: int,
    src: str,
    dst: str,
    errors: list[str],
) -> list[str] | None:
    prefix = f"proof-{round_no}-"
    route_id = str(route_id)
    if not route_id.startswith(prefix):
        errors.append(f"route_id 轮次前缀错误: {route_id!r}")
        return None
    rest = route_id[len(prefix) :]
    if not rest.startswith(src + "-") or not rest.endswith("-" + dst):
        errors.append(f"route_id 端点与 src/dst 不一致: {route_id!r}")
        return None
    middle = rest[len(src) + 1 : len(rest) - len(dst) - 1]
    if middle == "":
        nodes = [src, dst]
    else:
        switches = sorted(topo.switches, key=len, reverse=True)
        seg = _segment_nodes(middle, switches)
        if seg is None:
            errors.append(f"route_id 无法解析出交换机序列: {route_id!r}")
            return None
        nodes = [src] + seg + [dst]

    if len(set(nodes)) != len(nodes):
        errors.append(f"路径存在重复节点: {route_id!r}")
        return None
    bad = False
    for u, v in zip(nodes, nodes[1:]):
        if (u, v) not in topo.edges:
            errors.append(f"使用了未声明的边 {u}→{v}: {route_id!r}")
            bad = True
    return None if bad else nodes


def replay(topo: Topology, solution: dict, split: int | None = None) -> ReplayReport:
    """在本地按赛题约束回放方案，返回校验结果与通信耗时。"""
    errors: list[str] = []
    n = len(topo.ranks)
    operator = solution.get("operator")
    if operator not in ("all-gather", "all-reduce"):
        return ReplayReport(False, [f"未知算子: {operator!r}"])

    rounds = solution.get("rounds") or []
    if solution.get("round_count") != len(rounds):
        errors.append(
            f"round_count={solution.get('round_count')} 与 rounds 长度 {len(rounds)} 不一致"
        )

    if split is None:
        possession = solution.get("initial_possession") or {}
        first = topo.ranks[0]
        split = len(possession.get(first, [])) or 1

    chunk_bytes = topo.per_rank_bytes // split
    expected_chunks = [f"r{i}_c{s}" for i in range(n) for s in range(split)]
    chunks = list(solution.get("chunks") or [])
    if set(chunks) != set(expected_chunks):
        errors.append("chunks 与切分候选推导出的分片集合不一致")

    # 初始持有：必须来自任务规定的原始归属
    possession = solution.get("initial_possession") or {}
    if set(possession) != set(topo.ranks):
        errors.append("initial_possession 的 rank 集合与任务不一致")
    for i, rank in enumerate(topo.ranks):
        want = {f"r{i}_c{s}" for s in range(split)}
        got = set(possession.get(rank, []))
        if got != want:
            errors.append(f"initial_possession[{rank}] 非法: {sorted(got)}")

    # 状态：AllGather 用 chunk 集合，AllReduce 用 (slot -> 贡献者集合)
    poss: dict[str, set[str]] = {}
    contrib: dict[str, dict[int, frozenset[int]]] = {}
    if operator == "all-gather":
        for i, rank in enumerate(topo.ranks):
            poss[rank] = {f"r{i}_c{s}" for s in range(split)}
    else:
        for i, rank in enumerate(topo.ranks):
            contrib[rank] = {s: frozenset((i,)) for s in range(split)}

    sent: set = set()
    round_times_us: list[float] = []
    transfer_count = 0

    for r_idx, rnd in enumerate(rounds):
        round_no = rnd.get("round", r_idx + 1)
        edge_load: dict[tuple[str, str], int] = defaultdict(int)
        transfer_times: list[float] = []
        latency_max = 0.0
        pending_copy: dict[str, set[str]] = defaultdict(set)
        pending_reduce: dict[tuple[str, int], list[frozenset[int]]] = defaultdict(list)

        for t in rnd.get("transfers") or []:
            src, dst = str(t.get("src")), str(t.get("dst"))
            action = t.get("action")
            ranges = list(t.get("chunk_ranges") or [])
            transfer_count += 1
            if src not in poss and src not in contrib:
                errors.append(f"未知 src: {src}")
                continue
            if dst not in poss and dst not in contrib:
                errors.append(f"未知 dst: {dst}")
                continue
            if src == dst:
                errors.append(f"src 与 dst 相同: {src}")
                continue
            path = _resolve_route(
                topo, t.get("route_id", ""), round_no, src, dst, errors
            )
            if path is None:
                continue

            if operator == "all-gather" and action != "copy":
                errors.append(f"AllGather 只允许 copy，出现 {action!r}")
                continue
            if operator == "all-reduce" and action not in ("reduce", "copy"):
                errors.append(f"AllReduce 非法 action: {action!r}")
                continue

            if not ranges:
                errors.append(f"空 chunk_ranges: {t.get('route_id')!r}")
                continue

            # 正确性检查（全部基于本轮开始时的状态）
            if action == "copy":
                for name in ranges:
                    if name not in poss[src]:
                        errors.append(f"持有性违规: {src} 未持有 {name}")
                    if name in poss[dst] or name in pending_copy[dst]:
                        errors.append(f"接收唯一性违规: {dst} 重复接收 {name}")
                    key = (name, src, dst)
                    if key in sent:
                        errors.append(f"发送唯一性违规: {key}")
                    sent.add(key)
                    pending_copy[dst].add(name)
            else:  # reduce
                for expr in ranges:
                    parsed = _parse_expr(str(expr))
                    if parsed is None:
                        errors.append(f"无法解析规约表达式: {expr!r}")
                        continue
                    slot, cs = parsed
                    if len(cs) == 0:
                        cs = frozenset(range(n))
                    if slot >= split:
                        errors.append(f"槽位越界: {expr!r}")
                        continue
                    if max(cs, default=-1) >= n:
                        errors.append(f"贡献者编号越界: {expr!r}")
                        continue
                    if contrib[src].get(slot) != cs:
                        errors.append(f"持有性违规: {src} 未持有 {expr!r} 的完整集合")
                    cur = contrib[dst].get(slot, frozenset())
                    if cur & cs:
                        errors.append(
                            f"重复贡献: {dst} 的槽位 c{slot} 与 {expr!r} 有交集"
                        )
                    for other in pending_reduce[(dst, slot)]:
                        if other & cs:
                            errors.append(f"同轮重复贡献: 槽位 c{slot} 出现重叠集合")
                    key = (slot, cs, src, dst)
                    if key in sent:
                        errors.append(f"发送唯一性违规: {key}")
                    sent.add(key)
                    pending_reduce[(dst, slot)].append(cs)

            # 计时
            hops = len(path) - 2
            latency = topo.startup_us + hops * topo.switch_overhead_us
            t_us = latency
            bytes_ = len(ranges) * chunk_bytes
            for u, v in zip(path, path[1:]):
                bw = topo.edges[(u, v)]
                t_us += 1e6 * bytes_ / bw
                edge_load[(u, v)] += bytes_
            transfer_times.append(t_us)
            latency_max = max(latency_max, latency)

        # 应用本轮结果
        for dst, names in pending_copy.items():
            poss[dst] |= names
        for (dst, slot), cs_list in pending_reduce.items():
            merged = contrib[dst].get(slot, frozenset())
            for cs in cs_list:
                merged |= cs
            contrib[dst][slot] = merged

        if transfer_times:
            busiest = max(
                (1e6 * b / topo.edges[e] for e, b in edge_load.items()), default=0.0
            )
            round_times_us.append(max(max(transfer_times), latency_max + busiest))
        else:
            round_times_us.append(0.0)

    # 最终状态
    if operator == "all-gather":
        full = set(expected_chunks)
        for rank in topo.ranks:
            missing = full - poss[rank]
            if missing:
                errors.append(f"最终状态不完整: {rank} 缺少 {len(missing)} 个分片")
    else:
        full_set = frozenset(range(n))
        for rank in topo.ranks:
            for s in range(split):
                if contrib[rank].get(s) != full_set:
                    errors.append(f"最终状态不完整: {rank} 的 c{s} 贡献者集合不完整")

    return ReplayReport(
        ok=not errors,
        errors=errors,
        round_times_us=round_times_us,
        total_time_us=float(sum(round_times_us)),
        transfer_count=transfer_count,
    )
