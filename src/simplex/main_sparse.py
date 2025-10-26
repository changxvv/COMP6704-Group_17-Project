"""
Sparse-based entry point for Multi-Commodity Network Flow using linprog(HiGHS).

Builds A_eq/A_ub as scipy.sparse matrices to handle large instances without OOM.
This file is standalone and does not affect existing simplex/main.py behavior.
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from typing import Optional

from .parser import parse_sndlib, parse_lmcf_files


def build_sparse_mcf_lp(network, demands, capacity: Optional[float], use_sndlib_costs: bool = True):
    nodes = list(network.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    arcs = list(network.arcs)
    arc_index = {a: i for i, a in enumerate(arcs)}

    n_demands = len(demands)
    n_nodes = len(nodes)
    n_arcs = len(arcs)

    n_vars = n_demands * n_arcs
    n_eq = n_demands * n_nodes
    n_ub = n_arcs

    # Objective vector
    c = np.zeros(n_vars, dtype=float)
    for d in range(n_demands):
        for a, arc in enumerate(arcs):
            col = d * n_arcs + a
            if use_sndlib_costs and hasattr(network, "link_costs"):
                c[col] = network.link_costs.get(arc, 1.0)
            else:
                c[col] = 1.0

    # Precompute per-node outgoing/incoming arc indices to avoid scanning all arcs
    out_arc_indices = {n: [] for n in nodes}
    in_arc_indices = {n: [] for n in nodes}
    for a, (u, v) in enumerate(arcs):
        out_arc_indices[u].append(a)
        in_arc_indices[v].append(a)

    # Equality constraints (flow conservation) in COO (optimized)
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(n_eq, dtype=float)

    for d, demand in enumerate(demands):
        base_col = d * n_arcs
        base_row = d * n_nodes
        for n in nodes:
            row = base_row + node_index[n]
            # outgoing +1
            outs = out_arc_indices.get(n, [])
            if outs:
                eq_rows.extend([row] * len(outs))
                eq_cols.extend([base_col + idx for idx in outs])
                eq_data.extend([1.0] * len(outs))
            # incoming -1
            ins = in_arc_indices.get(n, [])
            if ins:
                eq_rows.extend([row] * len(ins))
                eq_cols.extend([base_col + idx for idx in ins])
                eq_data.extend([-1.0] * len(ins))

            if n == demand.source:
                b_eq[row] = demand.bandwidth
            elif n == demand.target:
                b_eq[row] = -demand.bandwidth
            else:
                b_eq[row] = 0.0

    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_eq, n_vars))

    # Inequality constraints (capacity) in COO
    ub_rows = []
    ub_cols = []
    ub_data = []
    b_ub = np.zeros(n_ub, dtype=float)

    for a, arc in enumerate(arcs):
        cap = getattr(network, "link_capacities", {}).get(arc, capacity)
        if cap is None:
            cap = 0.0
        b_ub[a] = float(cap)
        for d in range(n_demands):
            col = d * n_arcs + a
            ub_rows.append(a); ub_cols.append(col); ub_data.append(1.0)

    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_ub, n_vars))

    return c, A_eq, b_eq, A_ub, b_ub


def main():
    parser = argparse.ArgumentParser(description="Sparse MCNF solver (HiGHS)")
    parser.add_argument("instance", type=str, help="Path to instance file")
    parser.add_argument("--demand-file", type=str, help="Path to demand file (for LMCF)")
    parser.add_argument("--format", choices=["sndlib", "lmcf"], default="sndlib")
    parser.add_argument("--capacity", type=float, default=None, help="Uniform capacity fallback")
    parser.add_argument("--unit-costs", action="store_true", help="Use unit costs instead of file costs")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit (seconds)")
    args = parser.parse_args()

    inst = Path(args.instance)
    if not inst.exists():
        print(f"Error: instance not found: {inst}")
        sys.exit(1)

    if args.format == "lmcf":
        if not args.demand_file:
            print("Error: --demand-file is required for LMCF format")
            sys.exit(1)
        dem = Path(args.demand_file)
        if not dem.exists():
            print(f"Error: demand file not found: {dem}")
            sys.exit(1)
        network, demands, file_capacity = parse_lmcf_files(inst, dem)
    else:
        network, demands, file_capacity = parse_sndlib(inst)

    capacity = args.capacity if args.capacity is not None else file_capacity

    n_nodes = len(network.nodes)
    n_arcs = len(network.arcs)
    n_demands = len(demands)

    print("=" * 80)
    print("Sparse Multi-Commodity Flow (HiGHS)")
    print("=" * 80)
    print("\n【测试问题规模】")
    caps = list(getattr(network, 'link_capacities', {}).values())
    if caps:
        cap_min, cap_max = min(caps), max(caps)
    else:
        cap_min = cap_max = capacity if capacity is not None else 0.0
    total_demand = sum(d.bandwidth for d in demands) if demands else 0.0
    avg_demand = total_demand / n_demands if n_demands else 0.0
    print(f"  节点数: {n_nodes}")
    print(f"  边数: {n_arcs}")
    print(f"  需求数: {n_demands}")
    print(f"  容量范围: {cap_min:.2f} - {cap_max:.2f}")
    print(f"  总需求量: {total_demand:.2f}")
    print(f"  平均需求: {avg_demand:.2f}")
    print("\n" + "=" * 80 + "\n")

    t0 = time.perf_counter()
    c, A_eq, b_eq, A_ub, b_ub = build_sparse_mcf_lp(
        network, demands, capacity, use_sndlib_costs=not args.unit_costs
    )
    t1 = time.perf_counter()
    print(f"[sparse] Built LP (sparse) in {(t1 - t0):.2f}s")
    print(f"[sparse] Shapes: A_eq={A_eq.shape}, A_ub={A_ub.shape}, vars={len(c)}")
    print(f"[sparse] nnz: A_eq={A_eq.nnz}, A_ub={A_ub.nnz}")

    bounds = (0, None)
    options = {"disp": False}
    if args.time_limit is not None:
        options["time_limit"] = float(args.time_limit)

    print("Starting HiGHS (sparse)...")
    t2 = time.perf_counter()
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    t3 = time.perf_counter()

    print(f"[sparse] Solve time: {(t3 - t2):.2f}s (total {(t3 - t0):.2f}s)")
    print(f"[sparse] success={res.success}, status={res.status}, msg={res.message}")
    if hasattr(res, "nit"):
        print(f"[sparse] iterations={res.nit}")
    if hasattr(res, "fun") and res.fun is not None:
        print(f"[sparse] objective={res.fun:.6f}")

    if not res.success:
        sys.exit(2)

    print("\nDone.")


if __name__ == "__main__":
    main()


