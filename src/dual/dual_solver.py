import time
from typing import Dict, List, Tuple, Optional

from .data_models import Network, Demand
from .utils import dijkstra_shortest_path


def _to_key(u) -> str:
    """统一把节点 id 转成 str，避免 (int,str) 混用导致的 key mismatch。"""
    return str(u)


class DualSolver:
    """
    Dual Decomposition for arc-based MCNF with fractional (multi-route) repair.
    （略，注释同前）
    """

    def __init__(
        self,
        network: Network,
        time_limit: float = 300.0,
        max_iters: int = 800,
        alpha0: float = 100.0,
        alpha_decay: float = 0.9995,
        use_normalized_step: bool = True,
        enable_repair: bool = True,
        repair_iters: int = 1000,
        penalty_M: float = 20000.0,
        final_tol: float = 1e-6,
    ) -> None:
        self.network = network
        self.time_limit = time_limit
        self.max_iters = max_iters
        self.alpha0 = alpha0
        self.alpha_decay = alpha_decay
        self.use_normalized_step = use_normalized_step
        self.enable_repair = enable_repair
        self.repair_iters = repair_iters
        self.penalty_M = penalty_M
        self.final_tol = final_tol

        self.history: List[Dict[str, float]] = []

        self.base_costs: Dict[Tuple[str, str], float] = {}
        for e in network.edges.values():
            c = getattr(e, "init_weight", 1.0) or 1.0
            self.base_costs[(_to_key(e.u.id), _to_key(e.v.id))] = float(c)

        self.capacities: Dict[Tuple[str, str], float] = {
            (_to_key(e.u.id), _to_key(e.v.id)): float(e.capacity)
            for e in network.edges.values()
        }

        self.lmbd: Dict[Tuple[str, str], float] = {arc: 0.0 for arc in self.base_costs}
        self.demands_list: List[Demand] = list(network.demands.values())
        self.best_mix: Dict[int, List[Dict[str, float]]] = {}

        assert set(self.base_costs.keys()) == set(self.capacities.keys()), \
            "base_costs 与 capacities 的边键集合不一致"

    # ---------- helpers ----------

    def _weights(self, extra: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[Tuple[str, str], float]:
        if extra is None:
            return {e: self.base_costs[e] + self.lmbd[e] for e in self.base_costs}
        return {e: self.base_costs[e] + self.lmbd[e] + extra.get(e, 0.0) for e in self.base_costs}

    def _route_all_demands(self, weights: Dict[Tuple[str, str], float]):
        routes: Dict[int, List[List[str]]] = {}
        sp_costs: Dict[int, float] = {}
        for d_id, dem in enumerate(self.demands_list):
            nodes, dist = dijkstra_shortest_path(self.network, dem.origin, dem.destination, costs=weights)
            if nodes:
                routes[d_id] = [[_to_key(n.id) for n in nodes]]
                sp_costs[d_id] = float(dist)
        return routes, sp_costs

    def _usage_from_paths(self, paths: Dict[int, List[List[str]]]) -> Dict[Tuple[str, str], float]:
        usage: Dict[Tuple[str, str], float] = {e: 0.0 for e in self.base_costs}
        for d_id, plist in paths.items():
            dem = self.demands_list[d_id]
            flow = float(dem.quantity)
            for route in plist:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    usage[(u, v)] += flow
        return usage

    def _usage_from_mix(self, mix: Dict[int, List[Dict[str, float]]]) -> Dict[Tuple[str, str], float]:
        usage: Dict[Tuple[str, str], float] = {e: 0.0 for e in self.base_costs}
        for d_id, plist in mix.items():
            dem = self.demands_list[d_id]
            q = float(dem.quantity)
            for item in plist:
                route = item["path"]
                ratio = float(item["ratio"])
                if ratio <= 0 or not route:
                    continue
                flow = q * ratio
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    usage[(u, v)] += flow
        return usage

    def _dual_objective(self, sp_costs: Dict[int, float]) -> float:
        sum1 = 0.0
        for d_id, dist in sp_costs.items():
            d_k = float(self.demands_list[d_id].quantity)
            sum1 += d_k * dist
        sum2 = sum(self.capacities[e] * self.lmbd[e] for e in self.lmbd)
        return sum1 - sum2

    def _primal_cost_c(self, mix_or_paths) -> float:
        total = 0.0
        if not mix_or_paths:
            return 0.0
        first_val = next(iter(mix_or_paths.values()), [])
        if not first_val:
            return 0.0
        if isinstance(first_val, list):
            first_elem = first_val[0] if first_val else None
            # mix: list of dict
            if isinstance(first_elem, dict) and ('path' in first_elem):
                for d_id, plist in mix_or_paths.items():
                    q = float(self.demands_list[d_id].quantity)
                    for item in plist:
                        route = item["path"]
                        ratio = float(item["ratio"])
                        if not route or ratio <= 0.0:
                            continue
                        for i in range(len(route) - 1):
                            u, v = route[i], route[i + 1]
                            total += self.base_costs[(u, v)] * q * ratio
                return total
            # paths: list of list
            elif isinstance(first_elem, list):
                for d_id, plist in mix_or_paths.items():
                    q = float(self.demands_list[d_id].quantity)
                    for route in plist:
                        if not route:
                            continue
                        for i in range(len(route) - 1):
                            u, v = route[i], route[i + 1]
                            total += self.base_costs[(u, v)] * q
                return total
        # fallback
        try:
            for d_id, plist in mix_or_paths.items():
                q = float(self.demands_list[d_id].quantity)
                for item in plist:
                    route = item.get("path", [])
                    ratio = float(item.get("ratio", 0.0))
                    if not route or ratio <= 0.0:
                        continue
                    for i in range(len(route) - 1):
                        u, v = route[i], route[i + 1]
                        total += self.base_costs[(u, v)] * q * ratio
            return total
        except Exception:
            return 0.0

    # ---------- fractional repair ----------

    def _fractional_repair(self, init_paths: Dict[int, List[List[str]]]) -> Dict[int, List[Dict[str, float]]]:
        mix: Dict[int, List[Dict[str, float]]] = {}
        for d_id, plist in init_paths.items():
            assert len(plist) >= 1
            mix[d_id] = [{"path": plist[0], "ratio": 1.0}]

        def _normalize_ratios(lst):
            s = sum(item["ratio"] for item in lst)
            if s <= 0:
                lst[0]["ratio"] = 1.0
                s = 1.0
            for item in lst:
                item["ratio"] /= s

        eps0 = 0.5
        for _ in range(self.repair_iters):
            usage = self._usage_from_mix(mix)
            over = {e: usage[e] - self.capacities[e] for e in self.base_costs if usage[e] > self.capacities[e] + self.final_tol}
            if not over:
                break

            penalty = {e: self.penalty_M * float(over[e]) for e in over}
            weights = self._weights(penalty)
            alt: Dict[int, Optional[List[str]]] = {}
            for d_id, dem in enumerate(self.demands_list):
                nodes, _ = dijkstra_shortest_path(self.network, dem.origin, dem.destination, costs=weights)
                alt[d_id] = [_to_key(n.id) for n in nodes] if nodes else None

            changed = False
            for d_id, plist in list(mix.items()):
                cur_mix = plist
                alt_path = alt[d_id]
                if not alt_path:
                    continue

                def passes_over_edge(route):
                    for i in range(len(route) - 1):
                        if (route[i], route[i + 1]) in over:
                            return True
                    return False

                bad_idx = [idx for idx, item in enumerate(cur_mix) if passes_over_edge(item["path"])]
                if not bad_idx:
                    continue

                eps = eps0
                alt_idx = None
                for idx, item in enumerate(cur_mix):
                    if item["path"] == alt_path:
                        alt_idx = idx
                        break
                if alt_idx is None:
                    cur_mix.append({"path": alt_path, "ratio": 0.0})
                    alt_idx = len(cur_mix) - 1

                for idx in bad_idx:
                    if cur_mix[idx]["ratio"] <= 0.0:
                        continue
                    delta = min(cur_mix[idx]["ratio"], eps)
                    if delta <= 0:
                        continue
                    cur_mix[idx]["ratio"] -= delta
                    cur_mix[alt_idx]["ratio"] += delta
                    changed = True
                    eps *= 0.5

                _normalize_ratios(cur_mix)

            if not changed:
                break

        for d_id in mix:
            s = sum(item["ratio"] for item in mix[d_id])
            if s <= 0:
                mix[d_id][0]["ratio"] = 1.0
            else:
                for item in mix[d_id]:
                    item["ratio"] /= s
        return mix

    # ---------- main solve ----------

    def solve(self):
        start = time.time()
        alpha = self.alpha0

        best_paths: Optional[Dict[int, List[List[str]]]] = None
        best_obj = float("-inf")
        best_is_primal = False
        self.best_mix = {}

        self.history.clear()
        it = 0

        while it < self.max_iters and (time.time() - start) < self.time_limit:
            it += 1

            routes, sp_costs = self._route_all_demands(self._weights())
            usage = self._usage_from_paths(routes)
            grads = {e: usage[e] - self.capacities[e] for e in self.base_costs}
            total_violation = sum(max(0.0, g) for g in grads.values())
            max_violation = max([0.0] + [grads[e] for e in grads])
            overused_edges = sum(1 for e in grads if grads[e] > 0.0)

            feasible = (
                len(routes) == len(self.demands_list)
                and all(usage[e] <= self.capacities[e] + self.final_tol for e in self.base_costs)
            )

            dual_obj = self._dual_objective(sp_costs) if len(sp_costs) == len(self.demands_list) else float("-inf")
            primal_cost = self._primal_cost_c(routes) if routes else float("inf")

            if self.use_normalized_step:
                gnorm = (sum(g * g for g in grads.values())) ** 0.5 + 1e-12
                step = alpha / gnorm
            else:
                step = alpha

            elapsed = time.time() - start
            self.history.append(
                {
                    "iter": it,
                    "dual_obj": float(dual_obj),
                    "primal_cost": float(primal_cost),
                    "total_violation": float(total_violation),
                    "max_violation": float(max_violation),
                    "step": float(step),
                    "feasible": bool(feasible),
                    "covered_demands": int(len(routes)),
                    "num_demands": int(len(self.demands_list)),
                    "overused_edges": int(overused_edges),
                    "elapsed": float(elapsed),
                }
            )

            # 保存最好解
            if feasible:
                if (not best_is_primal) or (primal_cost < best_obj):
                    best_obj = primal_cost
                    best_paths = routes
                    best_is_primal = True
                # >>>> 新增：可行即早停 —— 直接退出对偶循环，进入分流修复 <<<<
                break
            else:
                if not best_is_primal and dual_obj > best_obj:
                    best_obj = dual_obj
                    best_paths = routes
                    best_is_primal = False

            # 对偶更新
            for e in self.lmbd:
                self.lmbd[e] = max(0.0, self.lmbd[e] + step * grads[e])
            alpha *= self.alpha_decay

        # 分流修复
        if self.enable_repair and best_paths is not None:
            mix = self._fractional_repair(best_paths)
            usage = self._usage_from_mix(mix)
            edge_ok = all(usage[e] <= self.capacities[e] + self.final_tol for e in self.base_costs)
            demand_ok = all(abs(sum(item["ratio"] for item in mix[d_id]) - 1.0) <= 1e-6 for d_id in mix)
            final_feasible = edge_ok and demand_ok and (len(mix) == len(self.demands_list))

            self.best_mix = mix
            if final_feasible:
                best_is_primal = True
                best_obj = self._primal_cost_c(mix)
                paths_compat: Dict[int, List[List[str]]] = {}
                for d_id, lst in mix.items():
                    paths_compat[d_id] = [it["path"] for it in lst if it["ratio"] > 1e-12]
                best_paths = paths_compat

        if best_paths is None:
            best_paths = {}

        elapsed = time.time() - start
        return best_paths, float(best_obj), it, float(elapsed)

    def get_best_mixture(self) -> Dict[int, List[Dict[str, float]]]:
        return self.best_mix
