"""
CLI entry point for Advanced Multi-Commodity Network Flow solver.
"""

import argparse
import sys
from pathlib import Path
import traceback

from .parser import parse_sndlib, parse_lmcf_files
from .simplex_solver import AdvancedSimplexSolver
from .utils import validate_solution, compute_objective


def main():
    """Main entry point for advanced MCNF solver."""
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Commodity Network Flow Solver"
    )
    parser.add_argument(
        "instance",
        type=str,
        help="Path to instance file"
    )
    parser.add_argument(
        "--demand-file",
        type=str,
        help="Path to demand file (required for LMCF format)"
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="Link capacity (overrides value from file)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300,
        help="Time limit in seconds (default: 300)"
    )
    parser.add_argument(
        "--unit-costs",
        action="store_true",
        help="Use unit costs instead of SNDlib routing costs"
    )
    parser.add_argument(
        "--method",
        choices=["primal", "dual", "primal_dual", "revised", "network"],
        default="primal_dual",
        help="Simplex method variant (default: primal_dual)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Numerical tolerance (default: 1e-8)"
    )
    parser.add_argument(
        "--format",
        choices=["sndlib", "lmcf"],
        default="sndlib",
        help="Input file format (default: sndlib)"
    )

    args = parser.parse_args()

    # Load instance
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance file not found: {args.instance}")
        sys.exit(1)

    try:
        if args.format == "sndlib":
            print(f"Loading SNDlib instance: {args.instance}")
            network, demands, file_capacity = parse_sndlib(instance_path)
        elif args.format == "lmcf":
            if not args.demand_file:
                print("Error: --demand-file is required for LMCF format")
                sys.exit(1)

            demand_path = Path(args.demand_file)
            if not demand_path.exists():
                print(f"Error: Demand file not found: {args.demand_file}")
                sys.exit(1)

            print(f"Loading LMCF instance: {args.instance}, {args.demand_file}")
            network, demands, file_capacity = parse_lmcf_files(instance_path, demand_path)
        else:
            print(f"Error: Unknown format: {args.format}")
            sys.exit(1)

    except Exception as e:
        print(f"Error parsing file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Use provided capacity or file capacity
    capacity = args.capacity if args.capacity is not None else file_capacity

    # Network summary
    node_count = len(network.nodes)
    arc_count = len(network.arcs)
    demand_count = len(demands)
    total_demand = sum(d.bandwidth for d in demands) if demands else 0.0
    avg_demand = total_demand / demand_count if demand_count > 0 else 0.0
    caps = list(getattr(network, 'link_capacities', {}).values())
    if caps:
        cap_min, cap_max = min(caps), max(caps)
    else:
        cap_min = cap_max = capacity if capacity is not None else 0.0

    # Header (中文)
    print("=" * 80)
    print("多商品流问题 (Multi-Commodity Flow) 求解器性能对比测试")
    print("=" * 80)
    print()
    print("【加载测试数据】")
    if args.format == "lmcf":
        print(f"  数据集: LMCF (file: {args.instance})")
        print(f"  网络文件: {args.instance}")
        print(f"  需求文件: {args.demand_file}")
    else:
        print(f"  数据集: SNDlib (file: {args.instance})")
    print()
    print("【测试问题规模】")
    print(f"  节点数: {node_count}")
    print(f"  边数: {arc_count}")
    print(f"  需求数: {demand_count}")
    print(f"  容量范围: {cap_min:.2f} - {cap_max:.2f}")
    print(f"  总需求量: {total_demand:.2f}")
    print(f"  平均需求: {avg_demand:.2f}")
    print()
    print("=" * 80)
    print()

    # Run optimization with advanced solver
    try:
        solver = AdvancedSimplexSolver(
            network=network,
            capacity=capacity,
            demands=demands,
            time_limit=args.time_limit,
            method=args.method,
            tolerance=args.tolerance,
            use_sndlib_costs=not args.unit_costs
        )

        solution = solver.solve()

    except Exception as e:
        print(f"Error during optimization: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Print solver block
    print()
    print("=" * 80)
    print(f"【{solver.method.replace('_', ' ').title()} 求解器】")
    print("-" * 80)
    print(f"[{solver.method}] 问题规模: {node_count}节点, {arc_count}边, {demand_count}需求")
    print(f"[{solver.method}] 容量范围: {cap_min:.2f} - {cap_max:.2f}")
    print(f"[{solver.method}] 开始求解...")
    print(f"[{solver.method}] 求解时间: {solution.time:.4f}s")
    print(f"[{solver.method}] 目标值 (Best Value): {solution.objective:.6f}")
    if solution.primal_residual is not None:
        print(f"[{solver.method}] 原始残差: {solution.primal_residual:.2e}")
    if solution.dual_residual is not None:
        print(f"[{solver.method}] 对偶残差: {solution.dual_residual:.2e}")
    if solution.complementarity is not None:
        print(f"[{solver.method}] 互补性: {solution.complementarity:.2e}")
    print(f"[{solver.method}] 收敛状态: {solution.status}")
    print(f"[{solver.method}] 迭代次数: {solution.iterations}")
    print(f"[{solver.method}] LP变量数: {solution.lp_var_count if solution.lp_var_count is not None else len(solver.lp_problem.c)}")
    if solution.termination_reason:
        print(f"[{solver.method}] 终止原因: {solution.termination_reason}")
    print()

    # Summary block
    print("=" * 80)
    print("【求解结果汇总】")
    print("=" * 80)
    print(f"{solver.method.replace('_', ' ').title()} 求解器:")
    print(f"  求解时间: {solution.time:.4f}秒")
    print(f"  目标值: {solution.objective:.6f}")
    print(f"  终止原因: {solution.status}")
    print(f"  收敛状态: {'✓ 成功' if solution.status in ('optimal', 'optimal_stagnated') else '✗ 未成功'}")
    print(f"  迭代次数: {solution.iterations}")
    print("  解的质量:")
    if solution.primal_residual is not None:
        print(f"    原始可行性残差: {solution.primal_residual:.2e} {'⚠' if solution.primal_residual > 1e-6 else '✓'}")
    if solution.dual_residual is not None:
        print(f"    对偶可行性残差: {solution.dual_residual:.2e} {'✓' if solution.dual_residual < 1e-6 else '⚠'}")
    if solution.complementarity is not None:
        print(f"    互补性: {solution.complementarity:.2e} {'✓' if solution.complementarity < 1e-6 else '⚠'}")
    print("=" * 80)
    print()

    # If optimal, validate solution
    if solution.status == "optimal":
        print("\n" + "=" * 60)
        print("Solution Validation")
        print("=" * 60)

        is_valid = validate_solution(solution, network, demands, capacity)
        if is_valid:
            print("✓ Solution is VALID")
        else:
            print("✗ Solution is INVALID")

        # Verify objective
        computed_obj = compute_objective(solution.allocation, demands, network)
        print(f"Computed objective: {computed_obj:.6f}")
        print(f"Solver objective: {solution.objective:.6f}")
        print(f"Difference: {abs(computed_obj - solution.objective):.6f}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Solution Statistics")
    print("=" * 60)

    print(f"Algorithm: {solution.algorithm}")
    print(f"Status: {solution.status}")
    print(f"Objective: {solution.objective:.6f}")
    print(f"Time: {solution.time:.4f}s")
    print(f"Iterations: {solution.iterations}")

    if solution.status == "optimal":
        total_paths = sum(len(paths) for paths in solution.allocation.values())
        covered_demands = len([d for d in solution.allocation if solution.allocation[d]])
        avg_paths = total_paths / covered_demands if covered_demands > 0 else 0

        print(f"Covered demands: {covered_demands}/{len(demands)}")
        print(f"Total paths used: {total_paths}")
        print(f"Average paths per demand: {avg_paths:.2f}")

        # Show some path examples
        if covered_demands > 0:
            print("\nSample path allocations:")
            shown = 0
            for demand_id in range(min(3, len(demands))):
                if demand_id in solution.allocation and solution.allocation[demand_id]:
                    demand = demands[demand_id]
                    paths_weights = solution.allocation[demand_id]
                    print(f"\n  Demand {demand_id} ({demand.source} → {demand.target}, "
                          f"bw={demand.bandwidth:.2f}):")
                    for idx, (path, weight) in enumerate(paths_weights[:2], 1):
                        nodes = [path.edges[0][0]] + [e[1] for e in path.edges]
                        path_str = " → ".join(nodes)
                        print(f"    Path {idx} (weight={weight:.4f}): {path_str} "
                              f"(cost: {path.cost:.2f})")
                    if len(paths_weights) > 2:
                        print(f"    ... and {len(paths_weights) - 2} more paths")
                    shown += 1
                    if shown >= 2:
                        break

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
