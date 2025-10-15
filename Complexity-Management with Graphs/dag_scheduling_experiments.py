import argparse
import math
import os
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt


# Data structures
@dataclass
class Graph:
    """Adjacency-list directed graph."""
    n: int
    adj: List[List[int]]  # edges u -> v for v in adj[u]


# DAG generation
def generate_random_dag(n: int, edge_prob: float, seed: Optional[int] = None) -> Graph:
    """
    Generate a random DAG by assigning a random topological order (a permutation)
    and only adding edges forward along that order with probability p.
    This guarantees acyclicity.
    """
    if seed is not None:
        random.seed(seed)

    order = list(range(n))
    random.shuffle(order)
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    adj = [[] for _ in range(n)]
    # Only allow edges from nodes earlier in the order to later nodes
    for u in range(n):
        for v in range(n):
            if pos[u] < pos[v]:  # forward edge only
                if random.random() < edge_prob:
                    adj[u].append(v)

    return Graph(n=n, adj=adj)


def add_random_cycles(g: Graph, num_back_edges: int, seed: Optional[int] = None) -> None:
    """
    Intentionally add 'back edges' to introduce cycles.
    We pick pairs (u, v) uniformly at random and add v -> u when u already reaches v (roughly),
    approximated by simply adding random edges that may create back-links.
    For simplicity, we’ll just add random directed edges (v -> u) that weren’t present.
    """
    if seed is not None:
        random.seed(seed)

    n = g.n
    added = 0
    attempts = 0
    existing = { (u, v) for u in range(n) for v in g.adj[u] }
    while added < num_back_edges and attempts < num_back_edges * 20:
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            attempts += 1
            continue
        # Add edge v -> u if not present; this is likely to create a back-link
        if (v, u) not in existing:
            g.adj[v].append(u)
            existing.add((v, u))
            added += 1
        attempts += 1


# Algorithms: Topological Sort & Cycle Detection
def topological_sort_kahn(g: Graph) -> Optional[List[int]]:
    """
    Kahn's algorithm. Returns a topological ordering if the graph is acyclic,
    otherwise returns None when a cycle is detected (i.e., not all nodes processed).
    Time: O(n + m). Space: O(n + m)
    """
    n = g.n
    indeg = [0] * n
    for u in range(n):
        for v in g.adj[u]:
            indeg[v] += 1

    from collections import deque
    q = deque([u for u in range(n) if indeg[u] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in g.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != n:
        return None  # cycle detected
    return order


def detect_cycle_dfs(g: Graph) -> bool:
    """
    DFS-based cycle detection in a directed graph.
    Returns True if there is a cycle, False otherwise.
    Time: O(n + m)
    """
    n = g.n
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    sys.setrecursionlimit(max(1000000, n + 10))

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in g.adj[u]:
            if color[v] == GRAY:
                return True  # back edge found
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return True
    return False


# Benchmarking helpers
@dataclass
class TrialResult:
    n: int
    m: int
    edge_prob: float
    topo_runtime_ms: float
    topo_memory_kb: float
    cycle_runtime_ms: float
    cycle_memory_kb: float
    had_cycle: bool


def count_edges(g: Graph) -> int:
    return sum(len(lst) for lst in g.adj)


def measure_once(n: int, edge_prob: float, with_cycles: bool, seed: Optional[int]) -> TrialResult:
    # Build graph
    g = generate_random_dag(n, edge_prob, seed)

    # Optionally add random cycles
    if with_cycles:
        # Add ~1% of n as back edges (tweakable)
        add_random_cycles(g, max(1, n // 100), seed=seed)

    # Topological sort — measure time + memory
    tracemalloc.start()
    t0 = time.perf_counter()
    topo = topological_sort_kahn(g)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    topo_ms = (t1 - t0) * 1e3
    topo_kb = peak / 1024.0

    # Cycle detection — measure time + memory
    tracemalloc.start()
    t0 = time.perf_counter()
    has_cycle = detect_cycle_dfs(g)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cycle_ms = (t1 - t0) * 1e3
    cycle_kb = peak / 1024.0

    return TrialResult(
        n=n,
        m=count_edges(g),
        edge_prob=edge_prob,
        topo_runtime_ms=topo_ms,
        topo_memory_kb=topo_kb,
        cycle_runtime_ms=cycle_ms,
        cycle_memory_kb=cycle_kb,
        had_cycle=(topo is None) or has_cycle
    )


def run_experiments(
    sizes: List[int],
    edge_prob: float,
    trials: int,
    with_cycles: bool,
    seed: Optional[int]
) -> List[TrialResult]:
    results: List[TrialResult] = []
    base_seed = seed if seed is not None else random.randrange(1 << 30)

    for i, n in enumerate(sizes):
        for t in range(trials):
            trial_seed = (base_seed + i * 1000 + t)
            res = measure_once(n=int(n), edge_prob=edge_prob, with_cycles=with_cycles, seed=trial_seed)
            results.append(res)
            print(f"[OK] n={res.n}, m={res.m}, p={edge_prob}, "
                  f"topo={res.topo_runtime_ms:.2f}ms ({res.topo_memory_kb:.0f}KB), "
                  f"cycle={res.cycle_runtime_ms:.2f}ms ({res.cycle_memory_kb:.0f}KB), "
                  f"had_cycle={res.had_cycle}")
    return results


# Plotting
def plot_scaling(results: List[TrialResult], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Aggregate by n (average over trials)
    from collections import defaultdict
    bucket: Dict[int, List[TrialResult]] = defaultdict(list)
    for r in results:
        bucket[r.n].append(r)

    xs = sorted(bucket.keys())
    def avg(arr): return sum(arr) / len(arr)

    topo_ms = [avg([r.topo_runtime_ms for r in bucket[n]]) for n in xs]
    cycle_ms = [avg([r.cycle_runtime_ms for r in bucket[n]]) for n in xs]
    topo_kb = [avg([r.topo_memory_kb for r in bucket[n]]) for n in xs]
    cycle_kb = [avg([r.cycle_memory_kb for r in bucket[n]]) for n in xs]
    edges = [avg([r.m for r in bucket[n]]) for n in xs]

    # Runtime vs n
    plt.figure()
    plt.plot(xs, topo_ms, marker='o', label='Topo sort (ms)')
    plt.plot(xs, cycle_ms, marker='o', label='Cycle detect (ms)')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'runtime_vs_n.png'), dpi=200)

    # Memory vs n
    plt.figure()
    plt.plot(xs, topo_kb, marker='o', label='Topo sort (KB)')
    plt.plot(xs, cycle_kb, marker='o', label='Cycle detect (KB)')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Peak memory (KB)')
    plt.title('Memory scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'memory_vs_n.png'), dpi=200)

    # Average edges vs n
    plt.figure()
    plt.plot(xs, edges, marker='o', label='Edges m')
    plt.xlabel('Number of nodes (n)')
    plt.ylabel('Average edges (m)')
    plt.title('Graph density scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'edges_vs_n.png'), dpi=200)

    print(f"Saved plots to: {outdir}")


# CLI
def parse_sizes(values: List[str]) -> List[int]:
    out = []
    for v in values:
        if v.lower().endswith('e3') or v.lower().endswith('e4') or v.lower().endswith('e5') or 'e' in v.lower():
            out.append(int(float(v)))
        else:
            out.append(int(v))
    return out


def main():
    ap = argparse.ArgumentParser(description="Variant 5 — DAG Scheduling Experiments")
    ap.add_argument("--sizes", nargs="+", default=["1000", "5000", "10000"],
                    help="Graph sizes (n). Accepts integers or scientific notation (e.g., 1e4).")
    ap.add_argument("--edge-prob", type=float, default=0.002,
                    help="Probability of a forward edge between a pair of nodes.")
    ap.add_argument("--trials", type=int, default=3, help="Trials per size.")
    ap.add_argument("--with-cycles", action="store_true",
                    help="If set, randomly introduce cycles to test detection.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument("--outdir", type=str, default="out_plots", help="Directory for plots.")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    results = run_experiments(
        sizes=sizes,
        edge_prob=args.edge_prob,
        trials=args.trials,
        with_cycles=args.with_cycles,
        seed=args.seed
    )
    plot_scaling(results, args.outdir)


if __name__ == "__main__":
    main()
