# AgentCC 最小求解 demo（拓扑 1）

给定拓扑输入 JSON，自动生成集合通信调度方案（`all-gather` / `all-reduce`），
并用内置的本地 Replay 按赛题计时口径校验正确性、计算通信耗时。

## 环境（uv）

```sh
uv sync          # 创建 .venv（只需标准库 + 开发用 pytest）
```

## 运行

```sh
# 默认：拓扑 1 + AllGather，solutions 输出到 out/
uv run solve.py
uv run solve.py --operator all-reduce --split 4
uv run solve.py --operator all-gather --split 1 --fanout 7 --out out/ag.json

# 只校验一份已有 solution
uv run solve.py --check out/topology1_all_gather_solution.json

# 自检
uv run pytest -q
```

## 结构

```
solve.py               CLI 入口：读拓扑 → 构造调度 → Replay 校验 → 输出 solution JSON
agentcc/topology.py    拓扑解析（rank/交换机/边/带宽/切分候选）+ 最短合法路径搜索
agentcc/planner.py     AllGather 递归加倍 / AllReduce butterfly，多路径并行（fanout）
agentcc/replay.py      本地 Replay：硬约束校验 + 按文档口径计时
tests/                 拓扑 1 跑通与负例自检
```

## 策略说明

- **路径**：全部由 BFS 在输入声明的有向边（`bidirectional` 自动补反向）上求解；
  中间节点只能是交换机，不按拓扑名称补全连接。
- **AllGather**：按 rank 下标二进制位递归加倍，共 `log2(N)` 轮；每对 rank 每轮
  交换全部已持有分片，同一逻辑发送按 `--fanout` 拆成多条互不共边的路径并行。
- **AllReduce**：butterfly 全规约，共 `log2(N)` 轮 `reduce`；第 k 轮每对 rank 合并
  贡献者集合互不相交的聚合分片（`r0_c0+r1_c0` 形式），7 轮后每个 rank 覆盖全部
  槽位与贡献者集合（`final_chunks` 为 `r0_c0+...+r127_c0` 形式）。
- **commands**：同一轮内的路径选择按共享边负载做贪心，尽量错开 L1/L2 链路。
- 输入的 `data_size`、`split_policy`、带宽、启动/交换机开销全部读取自拓扑文件；
  字符串 `1MB` 按规范换算为 `1,048,576` 字节，`GB/s` 按十进制换算。
