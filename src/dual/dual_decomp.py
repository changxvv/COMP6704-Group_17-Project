"""
Column Generation Controller (replaced by Dual Decomposition backend).
We preserve the class name and public API so main.py and CLI stay compatible.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from .data_models import Network
from .utils import reset_edge_capacity, reset_edge_weight
from .dual_solver import DualSolver

logger = logging.getLogger(__name__)


class DualDecomposition(DualSolver):
    def __init__(
        self,
        network: Network,
        max_iter,
        epsilon: float = 1e-6,
        output_folder: str = "./output/",
        M: float = 1e6
    ) -> None:
        self.network = network
        self.epsilon = epsilon
        self.max_iterations = max_iter
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.M = M

        self.iteration_history: List[Dict[str, Any]] = []
        self.solution_summary: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        logger.info("="*60)
        logger.info("Starting Dual Decomposition Solver (replacing Column Generation)")
        logger.info("="*60)

        start_time = time.time()

        reset_edge_capacity(self.network)
        reset_edge_weight(self.network)

        solver = DualSolver(network=self.network, max_iters=self.max_iterations)
        best_paths, objective, iterations, elapsed = solver.solve()
        self.iteration_history = list(solver.history)

        # 清空旧 routes
        for dem in self.network.demands.values():
            dem.routes.clear()

        # 写回解（支持分流）
        demand_list = list(self.network.demands.values())
        mix = solver.get_best_mixture()
        if mix:
            for d_id, items in mix.items():
                dem = demand_list[d_id]
                for it in items:
                    node_ids = it["path"]
                    ratio = it["ratio"]
                    route_nodes = [self.network.nodes[nid] for nid in node_ids]
                    dem.add_route(route_nodes, ratio)
                    flow = dem.quantity * ratio
                    for i in range(len(node_ids) - 1):
                        u, v = node_ids[i], node_ids[i + 1]
                        e_id = self.network.edge_dict.get((u, v))
                        if e_id is not None:
                            self.network.edges[e_id].add_flow(flow)
        else:
            for d_id, route_list in best_paths.items():
                dem = demand_list[d_id]
                node_ids = route_list[0]
                route_nodes = [self.network.nodes[nid] for nid in node_ids]
                dem.add_route(route_nodes, 1.0)
                for i in range(len(node_ids)-1):
                    u, v = node_ids[i], node_ids[i+1]
                    e_id = self.network.edge_dict.get((u, v))
                    if e_id is not None:
                        self.network.edges[e_id].add_flow(dem.quantity)

        # 计算最终可行性与指标
        tol = 1e-6
        edges = list(self.network.edges.values())
        edge_viols = [max(e.capacity_used - e.capacity, 0.0) for e in edges]
        total_violation = float(sum(edge_viols))
        max_violation = float(max([0.0] + edge_viols))
        overused_edges = int(sum(1 for v in edge_viols if v > tol))
        edge_ok = all(v <= tol for v in edge_viols)

        demand_ok = all(abs(sum(d.routes.values()) - 1.0) <= tol for d in self.network.demands.values())
        final_feasible = edge_ok and demand_ok

        # 解的质量附加指标
        best_dual = max([h.get('dual_obj', float('-inf')) for h in self.iteration_history] or [float('-inf')])
        # dual feasibility: λ >= 0，由投影保证；若非负则 residual=0
        lambdas = getattr(solver, "lmbd", {})
        min_lambda = min(lambdas.values()) if lambdas else 0.0
        dual_residual = max(0.0, -float(min_lambda))
        # complementarity: sum_e λ_e * max(usage_e - u_e, 0)
        # 注意 alt_dual_solver 中键使用 str
        lam_prod = 0.0
        for e in edges:
            key = (str(e.u.id), str(e.v.id))
            lam = float(lambdas.get(key, 0.0))
            over = max(e.capacity_used - e.capacity, 0.0)
            lam_prod += lam * over
        complementarity = float(lam_prod)

        status = "optimal" if final_feasible else "infeasible_or_not_converged"
        solve_time = time.time() - start_time

        # 汇总（供 save_results 与 main 打印）
        self.solution_summary = {
            "time": float(solve_time),
            "objective": float(objective),
            "status": status,
            "iterations": int(iterations),
            "primal_residual": float(total_violation),
            "dual_residual": float(dual_residual),
            "complementarity": float(complementarity),
            "best_dual": float(best_dual),
            "edge_ok": bool(edge_ok),
            "demand_ok": bool(demand_ok),
            "overused_edges": int(overused_edges),
            "max_violation": float(max_violation),
            "final_feasible": bool(final_feasible),
        }

        logger.info(f"Final feasibility: {final_feasible} (edge_ok={edge_ok}, demand_ok={demand_ok})")
        logger.info("="*60)
        logger.info(f"Solve complete. Objective: {objective:.4f}, Time: {elapsed:.2f}s, Iterations: {iterations}")
        logger.info("="*60)

        # 保存结果到文件
        self.save_results()

        # 返回给 main 用于打印
        return {
            'objective': objective,
            'solve_time': solve_time,
            'iterations': iterations,
            'is_feasible': final_feasible,
            'status': status,
            'primal_residual': total_violation,
            'dual_residual': dual_residual,
            'complementarity': complementarity,
        }

    def save_results(self) -> None:
        """Save summary, routes, flows, and dual-solver iteration history."""
        # ---- 新增：保存求解摘要 summary.txt ----
        summary_path = self.output_folder / "summary.txt"
        s = self.solution_summary or {}
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== Solution Summary ===\n")
            f.write(f"time_sec:           {s.get('time', float('nan')):.6f}\n")
            f.write(f"objective:          {s.get('objective', float('nan')):.6f}\n")
            f.write(f"status:             {s.get('status', 'unknown')}\n")
            f.write(f"iterations:         {s.get('iterations', 0)}\n")
            f.write(f"primal_residual:    {s.get('primal_residual', float('nan')):.6e}\n")
            f.write(f"dual_residual:      {s.get('dual_residual', float('nan')):.6e}\n")
            f.write(f"complementarity:    {s.get('complementarity', float('nan')):.6e}\n")
            f.write(f"best_dual:          {s.get('best_dual', float('nan')):.6f}\n")
            f.write(f"final_feasible:     {s.get('final_feasible', False)}\n")
            f.write(f"edge_ok:            {s.get('edge_ok', False)}\n")
            f.write(f"demand_ok:          {s.get('demand_ok', False)}\n")
            f.write(f"overused_edges:     {s.get('overused_edges', 0)}\n")
            f.write(f"max_violation:      {s.get('max_violation', 0.0):.6f}\n")

        # ---- 原有输出 ----
        # routes
        with open(self.output_folder / "routes.txt", 'w', encoding='utf-8') as f:
            f.write("demand_id\tratio\troute\n")
            for demand in self.network.demands.values():
                for route_hash, ratio in demand.routes.items():
                    f.write(f"{demand.id}\t{ratio:.6f}\t{route_hash}\n")

        # flows
        with open(self.output_folder / "flow.txt", 'w', encoding='utf-8') as f:
            f.write("edge_id\tsource\ttarget\tflow\tcapacity\n")
            for edge in self.network.edges.values():
                f.write(f"{edge.id}\t{edge.u.id}\t{edge.v.id}\t{edge.capacity_used:.6f}\t{edge.capacity:.6f}\n")

        # history
        with open(self.output_folder / "solver_history.txt", 'w', encoding='utf-8') as f:
            f.write("iter\tdual_obj\tprimal_cost\ttotal_violation\tmax_violation\tstep\t"
                    "feasible\tcovered_demands\tnum_demands\toverused_edges\telapsed\n")
            for rec in self.iteration_history:
                f.write(f"{int(rec.get('iter', 0))}\t"
                        f"{rec.get('dual_obj', float('nan')):.6f}\t"
                        f"{rec.get('primal_cost', float('nan')):.6f}\t"
                        f"{rec.get('total_violation', float('nan')):.6f}\t"
                        f"{rec.get('max_violation', float('nan')):.6f}\t"
                        f"{rec.get('step', float('nan')):.6f}\t"
                        f"{int(rec.get('feasible', False))}\t"
                        f"{int(rec.get('covered_demands', 0))}\t"
                        f"{int(rec.get('num_demands', 0))}\t"
                        f"{int(rec.get('overused_edges', 0))}\t"
                        f"{rec.get('elapsed', float('nan')):.6f}\n")

        logger.info("Results saved successfully")
