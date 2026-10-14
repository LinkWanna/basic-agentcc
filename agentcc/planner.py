"""调度方案构造（两种算子的最简可用策略）。

- AllGather：按 rank 下标二进制位做递归加倍（128 rank → 7 轮）。第 k 轮里
  每一对 rank 交换各自当前持有的全部分片；同一逻辑发送按 fanout 拆成多条
  互不共边的路径并行，降低单条链路瓶颈。
- AllReduce：butterfly 全规约（128 rank → 7 轮 reduce）。第 k 轮每一对 rank
  合并贡献者集合互不相交的聚合分片（每轮贡献者集合翻倍），7 轮后每个 rank
  都持有所有槽位的全集结果。同轮多个 slot 走不同路径。

所有路径都来自拓扑输入声明的边（BFS 最短路径），不按拓扑名称补全连接。
"""

from __future__ import annotations

import re
from collections import defaultdict

from .topology import Topology, mix32

_NAME_RE = re.compile(r"^r(\d+)_c(\d+)$")


def _chunk_key(name: str) -> tuple[int, int]:
    m = _NAME_RE.match(name)
    if not m:
        return (1 << 30, 0)
    return (int(m.group(1)), int(m.group(2)))


def _chunk_name(rank_idx: int, slot: int) -> str:
    return f"r{rank_idx}_c{slot}"


def _expr_name(contributors: frozenset[int], slot: int) -> str:
    """聚合分片表达式：槽位 + 贡献者集合（按 rank 升序规范化）。"""
    return "+".join(f"r{i}_c{slot}" for i in sorted(contributors))


def _log2_exact(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"demo 只支持 2 的幂个 rank，当前 ranks={n}")
    return n.bit_length() - 1


def _partition(items: list, groups: int) -> list[list]:
    """把 items 均分为至多 groups 段连续子列表（前面的段最多大 1）。"""
    groups = max(1, min(groups, len(items)))
    q, r = divmod(len(items), groups)
    out, pos = [], 0
    for g in range(groups):
        size = q + (1 if g < r else 0)
        out.append(items[pos : pos + size])
        pos += size
    return out


def _edges_of(path: list[str]) -> set[tuple[str, str]]:
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}


def _validate(topo: Topology, split: int) -> None:
    if split <= 0:
        raise ValueError(f"非法切分数: {split}")
    if split not in topo.split_candidates:
        raise ValueError(f"切分数 {split} 不在任务候选 {topo.split_candidates} 内")
    if topo.per_rank_bytes % split:
        raise ValueError(
            f"每个 rank 数据量 {topo.per_rank_bytes} 不能被切分数 {split} 整除"
        )


class _RoundRouter:
    """同一轮内全局挑选路径：尽量让并行传输走互不共边的链路。"""

    def __init__(self, topo: Topology) -> None:
        self.topo = topo
        self.loads: dict[tuple[str, str], float] = defaultdict(float)

    def emit(
        self,
        round_no: int,
        src: str,
        dst: str,
        groups: list[list[str]],
        chunk_bytes: int,
        seed: int,
        action: str,
    ) -> list[dict]:
        candidates = self.topo.shortest_paths(src, dst)
        if not candidates:
            raise RuntimeError(f"拓扑中不存在 {src} → {dst} 的合法路径")
        order = sorted(range(len(candidates)), key=lambda i: mix32(seed, i))
        chosen_edges: set[tuple[str, str]] = set()
        transfers: list[dict] = []
        for group in groups:
            size_bytes = len(group) * chunk_bytes
            best, best_score = None, None
            for idx in order:
                path = candidates[idx]
                edges = _edges_of(path)
                if edges & chosen_edges:
                    continue
                score = (
                    max(self.loads[e] + size_bytes for e in edges),
                    mix32(seed ^ 0x5BD1E995, idx),
                )
                if best_score is None or score < best_score:
                    best, best_score = path, score
            if best is None:  # 互不共边路径不够时，退化为复用一条候选路径
                best = candidates[order[len(transfers) % len(order)]]
            chosen_edges |= _edges_of(best)
            for e in _edges_of(best):
                self.loads[e] += size_bytes
            transfers.append(
                {
                    "action": action,
                    "chunk_ranges": list(group),
                    "src": src,
                    "dst": dst,
                    "route_id": f"proof-{round_no}-" + "-".join(best),
                }
            )
        return transfers


def _common_fields(
    topo: Topology, operator: str, chunks: list[str], possession
) -> dict:
    return {
        "operator": operator,
        "chunks": chunks,
        "initial_possession": possession,
        "initial_switch": list(topo.switches),
    }


# ------------------------------------------------------------------ AllGather
def plan_all_gather(topo: Topology, split: int = 1, fanout: int = 7) -> dict:
    """递归加倍 AllGather：每轮每个 rank 与距离 2^k 的 partner 合并持有集合。"""
    _validate(topo, split)
    n = len(topo.ranks)
    steps = _log2_exact(n)
    chunk_bytes = topo.per_rank_bytes // split

    chunks = [_chunk_name(i, s) for i in range(n) for s in range(split)]
    hold = [{_chunk_name(i, s) for s in range(split)} for i in range(n)]

    rounds: list[dict] = []
    for k in range(steps):
        router = _RoundRouter(topo)
        pairs = [(i, i | (1 << k)) for i in range(n) if not (i >> k) & 1]
        pending = []
        for i, j in pairs:
            for src, dst in ((i, j), (j, i)):
                groups = _partition(sorted(hold[src], key=_chunk_key), fanout)
                pending.append((src, dst, groups))
        pending.sort(key=lambda p: -sum(len(g) for g in p[2]))

        transfers: list[dict] = []
        for src, dst, groups in pending:
            seed = mix32((k + 1) * 1000003 + src, dst)
            transfers.extend(
                router.emit(
                    k + 1,
                    topo.ranks[src],
                    topo.ranks[dst],
                    groups,
                    chunk_bytes,
                    seed,
                    action="copy",
                )
            )

        for i, j in pairs:  # 轮结束后再统一更新持有集合
            union = hold[i] | hold[j]
            hold[i] = set(union)
            hold[j] = set(union)
        rounds.append({"round": k + 1, "transfers": transfers})

    solution = _common_fields(
        topo,
        "all-gather",
        chunks,
        {topo.ranks[i]: [_chunk_name(i, s) for s in range(split)] for i in range(n)},
    )
    solution["final_chunks"] = list(chunks)
    solution["round_count"] = steps
    solution["rounds"] = rounds
    return solution


# ------------------------------------------------------------------ AllReduce
def plan_all_reduce(topo: Topology, split: int = 1, fanout: int = 7) -> dict:
    """butterfly 全规约：每轮与距离 2^k 的 partner 交换并合并贡献者集合。"""
    _validate(topo, split)
    n = len(topo.ranks)
    steps = _log2_exact(n)
    chunk_bytes = topo.per_rank_bytes // split

    chunks = [_chunk_name(i, s) for i in range(n) for s in range(split)]
    # state[i][slot] = 该槽位当前聚合分片的贡献者集合
    state: list[dict[int, frozenset[int]]] = [
        {s: frozenset((i,)) for s in range(split)} for i in range(n)
    ]

    rounds: list[dict] = []
    for k in range(steps):
        router = _RoundRouter(topo)
        pairs = [(i, i | (1 << k)) for i in range(n) if not (i >> k) & 1]
        pending = []
        for i, j in pairs:
            for src, dst in ((i, j), (j, i)):
                entries = [(s, _expr_name(state[src][s], s)) for s in range(split)]
                groups = _partition(entries, fanout)
                pending.append((src, dst, groups))

        transfers: list[dict] = []
        for src, dst, groups in pending:
            seed = mix32((k + 1) * 1000033 + src, dst)
            transfers.extend(
                router.emit(
                    k + 1,
                    topo.ranks[src],
                    topo.ranks[dst],
                    [[e for _, e in g] for g in groups],
                    chunk_bytes,
                    seed,
                    action="reduce",
                )
            )

        for i, j in pairs:
            for s in range(split):
                union = state[i][s] | state[j][s]
                state[i][s] = union
                state[j][s] = union
        rounds.append({"round": k + 1, "transfers": transfers})

    full = frozenset(range(n))
    solution = _common_fields(
        topo,
        "all-reduce",
        chunks,
        {topo.ranks[i]: [_chunk_name(i, s) for s in range(split)] for i in range(n)},
    )
    solution["final_chunks"] = [_expr_name(full, s) for s in range(split)]
    solution["round_count"] = steps
    solution["rounds"] = rounds
    return solution
