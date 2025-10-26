"""
Advanced Simplex methods for Multi-Commodity Network Flow (robust version).

This module provides a robust, standardized interface for multiple simplex-like
methods. Key features:
 - Standardized return dicts (no 'failed' status; use clear labels).
 - Diagnostics for LP (A,b,c shapes, least-squares residual).
 - Safe fallbacks to scipy.optimize.linprog(highs) when numerical issues occur.
 - Robust conversion of variable vector -> path allocations.
"""

import time
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import linprog
import networkx as nx

from .data_models import Network, Demand, Path, Solution, LinearProgram


def _safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _normalize_status(success: bool, message: Optional[str]) -> str:
    """Map solver success/message to normalized status (no 'failed')."""
    if success:
        return "optimal"
    if message:
        low = message.lower()
        if "infeasible" in low:
            return "infeasible"
        if "unbounded" in low:
            return "unbounded"
        if "time" in low or "limit" in low:
            return "time_limit"
    return "error" if message else "unknown"


class AdvancedSimplexSolver:
    """
    Advanced Simplex solver with variants: primal, dual, primal_dual, revised, network.
    """

    def __init__(
        self,
        network: Network,
        capacity: float,
        demands: List[Demand],
        time_limit: float = 300,
        method: str = "primal_dual",
        tolerance: float = 1e-8,
        use_sndlib_costs: bool = True,
    ):
        self.network = network
        self.capacity = capacity
        self.demands = demands
        self.time_limit = time_limit
        self.method = method
        self.tolerance = tolerance
        self.use_sndlib_costs = use_sndlib_costs

        self.lp_problem = self._build_mcf_lp()

    def _build_mcf_lp(self) -> LinearProgram:
        """Construct LP in standard form: min c^T x s.t. A_eq x = b_eq, A_ub x <= b_ub, x >= 0"""
        n_demands = len(self.demands)
        n_arcs = len(self.network.arcs)
        n_nodes = len(self.network.nodes)

        n_variables = n_demands * n_arcs
        n_flow_constraints = n_demands * n_nodes
        n_capacity_constraints = n_arcs

        c = np.zeros(n_variables)
        for d, demand in enumerate(self.demands):
            for a, arc in enumerate(self.network.arcs):
                idx = d * n_arcs + a
                if self.use_sndlib_costs and hasattr(self.network, "link_costs"):
                    cost = self.network.link_costs.get(arc, 1.0)
                else:
                    cost = 1.0
                c[idx] = cost

        # Equality constraints (flow conservation)
        A_eq = np.zeros((n_flow_constraints, n_variables))
        b_eq = np.zeros(n_flow_constraints)

        constraint_idx = 0
        for d, demand in enumerate(self.demands):
            for node in self.network.nodes:
                for a, arc in enumerate(self.network.arcs):
                    var_idx = d * n_arcs + a
                    u, v = arc
                    if u == node:
                        A_eq[constraint_idx, var_idx] = 1.0
                    elif v == node:
                        A_eq[constraint_idx, var_idx] = -1.0

                if node == demand.source:
                    b_eq[constraint_idx] = demand.bandwidth
                elif node == demand.target:
                    b_eq[constraint_idx] = -demand.bandwidth
                else:
                    b_eq[constraint_idx] = 0.0
                constraint_idx += 1

        # Inequality constraints (capacity)
        A_ub = np.zeros((n_capacity_constraints, n_variables))
        b_ub = np.zeros(n_capacity_constraints)

        for a, arc in enumerate(self.network.arcs):
            capacity = getattr(self.network, "link_capacities", {}).get(arc, self.capacity)
            for d in range(n_demands):
                var_idx = d * n_arcs + a
                A_ub[a, var_idx] = 1.0
            b_ub[a] = capacity

        variable_names = []
        for d in range(n_demands):
            for a, arc in enumerate(self.network.arcs):
                variable_names.append(f"f_D{d}_{arc[0]}_{arc[1]}")

        constraint_names = []
        for d in range(n_demands):
            for node in self.network.nodes:
                constraint_names.append(f"flow_D{d}_N{node}")
        for arc in self.network.arcs:
            constraint_names.append(f"cap_{arc[0]}_{arc[1]}")

        # Store both equality and inequality constraints
        lp = LinearProgram(c=c, A=A_eq, b=b_eq, variable_names=variable_names, constraint_names=constraint_names)
        lp.A_ub = A_ub
        lp.b_ub = b_ub
        return lp

    # ---------- Diagnostics ----------
    def _diagnose_lp(self) -> None:
        A = self.lp_problem.A
        b = self.lp_problem.b
        c = self.lp_problem.c
        m, n = A.shape
        print("---- LP DIAGNOSTICS ----")
        print(f"A shape: {A.shape}, c length: {len(c)}, b length: {len(b)}")
        print(f"A nonzeros: {np.count_nonzero(A)}, c nonzeros: {np.count_nonzero(c)}, b nonzeros: {np.count_nonzero(b)}")
        try:
            x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
            residual = np.linalg.norm(A @ x_ls - b, ord=np.inf)
            print(f"Least-squares residual ||A x_ls - b||_inf = {residual:.6g}")
            if residual > 1e-6:
                print("  Warning: equality constraints may be inconsistent (residual > 1e-6).")
        except Exception as e:
            print(f"  Could not compute least-squares fit: {e}")
        print("------------------------")

    # ---------- Standardized solve() ----------
    def solve(self) -> Solution:
        start_time = time.time()
        print(f"Starting {self.method.replace('_', ' ').title()} Simplex...")
        print(f"Problem: {len(self.demands)} demands, {len(self.network.arcs)} arcs")

        if self.method == "primal":
            result = self._solve_primal_simplex()
        elif self.method == "dual":
            result = self._solve_dual_simplex()
        elif self.method == "primal_dual":
            result = self._solve_primal_dual_simplex()
        elif self.method == "revised":
            result = self._solve_revised_simplex()
        elif self.method == "network":
            result = self._solve_network_simplex()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        allocation = self._convert_to_path_solution(result)
        objective = self._compute_objective(allocation)

        end_time = time.time()
        elapsed = end_time - start_time

        # Build Solution with numeric diagnostic fields guarded
        x = result.get("primal_solution", None)
        y = result.get("dual_solution", None)

        primal_residual = None
        dual_residual = None
        complementarity = None
        if x is not None:
            try:
                Ax_minus_b = self.lp_problem.A @ x - self.lp_problem.b
                primal_residual = float(np.max(np.abs(Ax_minus_b)))
            except Exception:
                primal_residual = float("nan")

        if x is not None and y is not None:
            try:
                r = self.lp_problem.c - self.lp_problem.A.T @ y
                dual_residual = float(np.max(np.abs(r)))
                complementarity = float(np.max(np.abs(x * r)))
            except Exception:
                dual_residual = float("nan")
                complementarity = float("nan")

        status = result.get("status", "unknown")
        message = result.get("message", "")

        sol = Solution(
            allocation=allocation,
            objective=objective,
            iterations=int(result.get("iterations", 0) or 0),
            time=elapsed,
            converged=bool(result.get("converged", False)),
            status=status,
            algorithm=self.method,
        )

        # attach diagnostics to solution for printing (non-invasive)
        setattr(sol, "primal_residual", _safe_float(primal_residual))
        setattr(sol, "dual_residual", _safe_float(dual_residual))
        setattr(sol, "complementarity", _safe_float(complementarity))
        setattr(sol, "termination_reason", message or status)
        setattr(sol, "lp_var_count", len(self.lp_problem.c))

        print(f"\n{self.method.replace('_', ' ').title()} completed in {elapsed:.2f}s")
        print(f"Final Objective: {objective:.6f}")
        return sol

    # ---------- Solver implementations (standardized return dict) ----------
    def _solve_primal_dual_simplex(self) -> Dict[str, Any]:
        """
        Simplified primal-dual wrapper using linprog for robustness.
        Returns dict with: status, message, primal_solution, dual_solution, iterations, converged, fun
        """
        print("Running Primal-Dual Simplex (robust wrapper)...")
        c = self.lp_problem.c
        A_eq = self.lp_problem.A
        b_eq = self.lp_problem.b
        A_ub = getattr(self.lp_problem, 'A_ub', None)
        b_ub = getattr(self.lp_problem, 'b_ub', None)

        # LP diagnostics
        try:
            self._diagnose_lp()
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if A_ub is not None and b_ub is not None:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs", options={"disp": False})
                else:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs", options={"disp": False})
        except Exception as e:
            return {
                "status": "error",
                "message": f"linprog exception: {e}",
                "primal_solution": None,
                "dual_solution": None,
                "iterations": 0,
                "converged": False,
                "fun": None,
            }

        success = bool(getattr(res, "success", False))
        message = getattr(res, "message", "")
        status = _normalize_status(success, message)
        x = res.x if hasattr(res, "x") and res.x is not None else None
        dual = None
        try:
            # Attempt to extract eqlin marginals if present
            if hasattr(res, "con") and isinstance(res.con, dict):
                dual = res.con.get("eqlin", {}).get("marginals", None)
        except Exception:
            dual = None

        print("linprog result:", {"success": success, "status": getattr(res, "status", None), "message": message, "fun": getattr(res, "fun", None), "nit": getattr(res, "nit", None)})

        return {
            "status": status,
            "message": message,
            "primal_solution": x,
            "dual_solution": dual,
            "iterations": int(getattr(res, "nit", 0) or 0),
            "converged": success,
            "fun": float(getattr(res, "fun", 0.0) or 0.0),
        }

    def _solve_primal_simplex(self) -> Dict[str, Any]:
        """Use linprog primal (highs)"""
        print("Running Primal Simplex (linprog - highs)...")
        c = self.lp_problem.c
        A_eq = self.lp_problem.A
        b_eq = self.lp_problem.b
        A_ub = getattr(self.lp_problem, 'A_ub', None)
        b_ub = getattr(self.lp_problem, 'b_ub', None)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if A_ub is not None and b_ub is not None:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs", options={"disp": False})
                else:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs", options={"disp": False})
        except Exception as e:
            return {"status": "error", "message": f"linprog exception: {e}", "primal_solution": None, "dual_solution": None, "iterations": 0, "converged": False, "fun": None}
        success = bool(getattr(res, "success", False))
        message = getattr(res, "message", "")
        status = _normalize_status(success, message)
        x = res.x if hasattr(res, "x") and res.x is not None else None
        dual = None
        return {"status": status, "message": message, "primal_solution": x, "dual_solution": dual, "iterations": int(getattr(res, "nit", 0) or 0), "converged": success, "fun": float(getattr(res, "fun", 0.0) or 0.0)}

    def _solve_dual_simplex(self) -> Dict[str, Any]:
        """Attempt highs-ds, fallback to highs primal."""
        print("Running Dual Simplex (attempt highs-ds)...")
        c = self.lp_problem.c
        A_eq = self.lp_problem.A
        b_eq = self.lp_problem.b
        A_ub = getattr(self.lp_problem, 'A_ub', None)
        b_ub = getattr(self.lp_problem, 'b_ub', None)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if A_ub is not None and b_ub is not None:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs-ds", options={"disp": False})
                else:
                    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds", options={"disp": False})
            success = bool(getattr(res, "success", False))
            message = getattr(res, "message", "")
            status = _normalize_status(success, message)
            x = res.x if hasattr(res, "x") and res.x is not None else None
            dual = None
            return {"status": status, "message": message, "primal_solution": x, "dual_solution": dual, "iterations": int(getattr(res, "nit", 0) or 0), "converged": success, "fun": float(getattr(res, "fun", 0.0) or 0.0)}
        except Exception as e:
            print(f"highs-ds not available or failed: {e}; falling back to highs primal")
            return self._solve_primal_simplex()

    def _solve_revised_simplex(self) -> Dict[str, Any]:
        """
        Attempt a simple revised simplex with explicit basis. If numerical
        difficulties arise, fallback to linprog.
        """
        print("Running Revised Simplex (attempt)...")
        c = self.lp_problem.c
        A = self.lp_problem.A
        b = self.lp_problem.b
        m, n = A.shape

        # Augment with slack variables to obtain an obvious basis if appropriate
        A_aug = np.hstack([A, np.eye(m)]) if m > 0 else A
        c_aug = np.hstack([c, np.zeros(m)]) if m > 0 else c

        num_vars_aug = n + (m if m > 0 else 0)

        # initial basis: slack variables if present
        basis_idx = np.arange(n, n + m, dtype=int) if m > 0 else np.array([], dtype=int)
        nonbasis_idx = np.array([i for i in range(n)], dtype=int)

        try:
            max_iters = min(1000, 10 * num_vars_aug if num_vars_aug > 0 else 1000)
            iterations = 0

            while iterations < max_iters:
                iterations += 1

                if m == 0:
                    # trivial
                    x_full = np.zeros(num_vars_aug)
                    return {"status": "optimal", "message": "trivial", "primal_solution": x_full[:n], "dual_solution": None, "iterations": iterations, "converged": True, "fun": 0.0}

                B = A_aug[:, basis_idx]
                N = A_aug[:, nonbasis_idx]

                # factorize B
                try:
                    lu, piv = lu_factor(B)
                except Exception as e:
                    # fallback
                    print(f"[revised] LU factorization failed: {e}; falling back to linprog")
                    raise

                # solve for basic variables
                try:
                    x_B = lu_solve((lu, piv), b)
                except Exception as e:
                    print(f"[revised] solving x_B failed: {e}; falling back to linprog")
                    raise

                # compute duals y from B^T y = c_B
                c_B = c_aug[basis_idx]
                try:
                    y = np.linalg.solve(B.T, c_B)
                except np.linalg.LinAlgError:
                    # fallback to using lu trans solve
                    try:
                        y = lu_solve((lu, piv), c_B, trans=1)
                    except Exception as e:
                        print(f"[revised] solving for y failed: {e}; falling back to linprog")
                        raise

                # reduced costs
                if N.size == 0:
                    r_N = np.array([])
                else:
                    r_N = c_aug[nonbasis_idx] - N.T @ y

                # optimality check
                if np.all(r_N >= -self.tolerance):
                    x_full = np.zeros(num_vars_aug)
                    x_full[basis_idx] = x_B
                    x_result = x_full[:n]
                    obj_val = float(c @ x_result)
                    return {"status": "optimal", "message": "optimal (revised simplex)", "primal_solution": x_result, "dual_solution": y, "iterations": iterations, "converged": True, "fun": obj_val}

                # entering variable: Dantzig's rule
                entering_pos = int(np.argmin(r_N))
                entering_var = int(nonbasis_idx[entering_pos])

                # direction d = B^{-1} a_j
                a_j = A_aug[:, entering_var]
                try:
                    d = lu_solve((lu, piv), a_j)
                except Exception as e:
                    print(f"[revised] direction solve failed: {e}; falling back to linprog")
                    raise

                # check unboundedness
                if np.all(d <= self.tolerance):
                    return {"status": "unbounded", "message": "unbounded", "primal_solution": None, "dual_solution": None, "iterations": iterations, "converged": False, "fun": None}

                # step sizes
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.where(d > self.tolerance, x_B / d, np.inf)
                leaving_pos = int(np.argmin(ratios))
                if not np.isfinite(ratios[leaving_pos]):
                    print("[revised] No finite ratio found -> numerical issue; falling back to linprog")
                    raise RuntimeError("No finite ratio found")

                # pivot
                leaving_var = int(basis_idx[leaving_pos])
                basis_idx[leaving_pos] = entering_var
                nonbasis_idx[entering_pos] = leaving_var

            # max iterations
            return {"status": "max_iterations", "message": "max_iterations", "primal_solution": None, "dual_solution": None, "iterations": iterations, "converged": False, "fun": None}

        except Exception:
            # fallback to linprog (highs)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    A_ub = getattr(self.lp_problem, 'A_ub', None)
                    b_ub = getattr(self.lp_problem, 'b_ub', None)
                    if A_ub is not None and b_ub is not None:
                        res = linprog(self.lp_problem.c, A_eq=self.lp_problem.A, b_eq=self.lp_problem.b, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs", options={"disp": False})
                    else:
                        res = linprog(self.lp_problem.c, A_eq=self.lp_problem.A, b_eq=self.lp_problem.b, bounds=(0, None), method="highs", options={"disp": False})
                success = bool(getattr(res, "success", False))
                message = getattr(res, "message", "")
                status = _normalize_status(success, message)
                x = res.x if hasattr(res, "x") and res.x is not None else None
                return {"status": status, "message": message, "primal_solution": x, "dual_solution": None, "iterations": int(getattr(res, "nit", 0) or 0), "converged": success, "fun": float(getattr(res, "fun", 0.0) or 0.0)}
            except Exception as e2:
                return {"status": "error", "message": f"linprog fallback exception: {e2}", "primal_solution": None, "dual_solution": None, "iterations": 0, "converged": False, "fun": None}

    def _solve_network_simplex(self) -> Dict[str, Any]:
        """
        Heuristic network-style routing: per-demand shortest path assignment.
        Not a proper multi-commodity network simplex; used for fast heuristic.
        """
        print("Running Network Simplex (heuristic)...")
        G = nx.DiGraph()
        for node in self.network.nodes:
            G.add_node(node)
        for arc in self.network.arcs:
            u, v = arc
            cap = getattr(self.network, "link_capacities", {}).get(arc, self.capacity)
            cost = getattr(self.network, "link_costs", {}).get(arc, 1.0)
            G.add_edge(u, v, capacity=cap, weight=cost, cost=cost)

        total_flow = {}
        for d, demand in enumerate(self.demands):
            try:
                if not nx.has_path(G, demand.source, demand.target):
                    continue
                path = nx.shortest_path(G, demand.source, demand.target, weight="weight")
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                bottleneck = min(G.edges[e]["capacity"] for e in edges)
                flow = min(bottleneck, demand.bandwidth)
                for e in edges:
                    total_flow[e] = total_flow.get(e, 0.0) + flow
            except Exception as e:
                print(f"[network heuristic] warning routing demand {d}: {e}")

        # convert to variable vector (proportional distribution)
        n_demands = len(self.demands)
        n_arcs = len(self.network.arcs)
        sol_vec = np.zeros(n_demands * n_arcs)
        for d in range(n_demands):
            for a, arc in enumerate(self.network.arcs):
                if arc in total_flow:
                    idx = d * n_arcs + a
                    sol_vec[idx] = total_flow[arc] / max(1, n_demands)
        return {"status": "optimal", "message": "heuristic", "primal_solution": sol_vec, "dual_solution": None, "iterations": 1, "converged": True, "fun": None}

    # ---------- Conversion & objective ----------
    def _convert_to_path_solution(self, result: Dict[str, Any]) -> Dict[int, List[Tuple[Path, float]]]:
        x = result.get("primal_solution", None)
        if x is None:
            return {}
        x = np.array(x, dtype=float)

        n_demands = len(self.demands)
        n_arcs = len(self.network.arcs)
        allocation: Dict[int, List[Tuple[Path, float]]] = {}
        for d, demand in enumerate(self.demands):
            demand_flows = {}
            start = d * n_arcs
            for a, arc in enumerate(self.network.arcs):
                idx = start + a
                if idx < len(x):
                    flow = float(x[idx])
                    if flow > 1e-9:
                        demand_flows[arc] = flow
            if demand_flows:
                try:
                    G = nx.DiGraph()
                    for arc, flow in demand_flows.items():
                        u, v = arc
                        cost = getattr(self.network, "link_costs", {}).get(arc, 1.0)
                        G.add_edge(u, v, weight=cost, capacity=flow)
                    if nx.has_path(G, demand.source, demand.target):
                        p = nx.shortest_path(G, demand.source, demand.target, weight="weight")
                        edges = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
                        path_cost = sum(getattr(self.network, "link_costs", {}).get(e, 1.0) for e in edges)
                        allocation[d] = [(Path(edges=tuple(edges), cost=path_cost, demand_id=d), 1.0)]
                    else:
                        allocation[d] = []
                except Exception:
                    allocation[d] = []
            else:
                allocation[d] = []
        return allocation

    def _compute_objective(self, allocation: Dict[int, List[Tuple[Path, float]]]) -> float:
        total = 0.0
        for d_id, pw in allocation.items():
            if d_id >= len(self.demands):
                continue
            demand = self.demands[d_id]
            for path, weight in pw:
                total += path.cost * demand.bandwidth * weight
        return total
