"""拓扑输入解析与最短合法路径搜索。

只依赖 Python 标准库。所有参数都从拓扑输入文件读取，不按拓扑名称推断连接、
带宽或路径（赛题硬性要求）。
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

_DATA_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B\s*$", re.IGNORECASE)
_BW_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B/s\s*$", re.IGNORECASE)
_DATA_SCALE = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_BW_SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_data_size(value: object) -> int:
    """解析 data_size（如 "1MB"）。规范约定数据量字符串 MB 按 2^20 字节。"""
    if isinstance(value, bool):
        raise ValueError(f"无法解析 data_size: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    m = _DATA_RE.match(str(value))
    if not m:
        raise ValueError(f"无法解析 data_size: {value!r}")
    num, unit = float(m.group(1)), m.group(2).upper()
    return int(num * _DATA_SCALE[unit])


def parse_bandwidth(value: object) -> float:
    """解析边带宽（如 "56GB/s"），GB/s、MB/s 按十进制换算，返回字节/秒。"""
    if isinstance(value, bool):
        raise ValueError(f"无法解析带宽: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    m = _BW_RE.match(str(value))
    if not m:
        raise ValueError(f"无法解析带宽: {value!r}")
    num, unit = float(m.group(1)), m.group(2).upper()
    return num * _BW_SCALE[unit]


def mix32(a: int, b: int) -> int:
    """确定性 32 位混合哈希（不要用内置 hash()，其跨进程随机）。"""
    x = (a * 0x9E3779B1 + b * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return x


@dataclass
class Topology:
    """拓扑 1/2/3/4 的通用解析结果（仅使用输入文件声明的字段）。"""

    ranks: list[str]
    switches: list[str]
    edges: dict[tuple[str, str], float]  # 有向 (u, v) -> 字节/秒；双向边已补反向
    startup_us: float
    switch_overhead_us: float
    per_rank_bytes: int
    split_candidates: list[int]
    allow_copy: bool = True
    allow_detour: bool = False
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._switch_set = set(self.switches)
        self._path_cache: dict[tuple[str, str], list[list[str]]] = {}
        adj: dict[str, set[str]] = {}
        for u, v in self.edges:
            adj.setdefault(u, set()).add(v)
        self.adj = {u: sorted(vs) for u, vs in adj.items()}

    # ------------------------------------------------------------------ 解析
    @classmethod
    def load(cls, path: str | Path) -> "Topology":
        with open(path, encoding="utf-8") as f:
            return cls.parse(json.load(f))

    @classmethod
    def parse(cls, data: dict) -> "Topology":
        edges: dict[tuple[str, str], float] = {}
        for e in data["edges"]:
            bw = parse_bandwidth(e["bw"])
            edges[(e["src"], e["dst"])] = bw
            if e.get("bidirectional"):
                edges[(e["dst"], e["src"])] = bw
        policy = data.get("split_policy") or {}
        candidates = (
            policy.get("candidates split")
            or policy.get("candidates_split")
            or policy.get("candidates")
            or [1]
        )
        return cls(
            ranks=list(data["ranks"]),
            switches=list(data["switches"]),
            edges=edges,
            startup_us=float(data.get("communication_startup_time_us", 0.0)),
            switch_overhead_us=float(data.get("switch_processing_overhead_us", 0.0)),
            per_rank_bytes=parse_data_size(data["data_size"]["per_rank_bytes"]),
            split_candidates=[int(c) for c in candidates],
            allow_copy=bool(data.get("allow_copy", True)),
            allow_detour=bool(data.get("allow_detour", False)),
            raw=data,
        )

    # ------------------------------------------------------------ 路径搜索
    def shortest_paths(self, src: str, dst: str, cap: int = 160) -> list[list[str]]:
        """返回 src→dst 的全部最短合法路径（最多 cap 条，含节点名序列）。

        规则：中间节点只能是输入声明的交换机；不允许把其他 rank 当过路节点
        （rank 中继必须拆成跨轮 transfer）。只使用输入声明的边与方向。
        """
        key = (src, dst)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached

        dist: dict[str, int] = {src: 0}
        queue = deque([src])
        switches = self._switch_set
        while queue:
            u = queue.popleft()
            if u == dst or (u != src and u not in switches):
                continue
            for v in self.adj.get(u, ()):
                if v == src or v in dist:
                    continue
                if v == dst or v in switches:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        paths: list[list[str]] = []
        if dst in dist:

            def dfs(u: str, path: list[str], seen: set[str]) -> None:
                if len(paths) >= cap:
                    return
                if u == dst:
                    paths.append(list(path))
                    return
                for v in self.adj.get(u, ()):
                    if v in seen or dist.get(v) != dist[u] + 1:
                        continue
                    seen.add(v)
                    path.append(v)
                    dfs(v, path, seen)
                    path.pop()
                    seen.discard(v)

            dfs(src, [src], {src})

        self._path_cache[key] = paths
        return paths
