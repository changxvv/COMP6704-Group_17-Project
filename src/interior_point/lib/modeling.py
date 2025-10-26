from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable


@dataclass
class LPStandardForm:
    A_eq: List[List[float]]
    b_eq: List[float]
    c: List[float]
    var_names: List[str]


class LpProblem:
    def __init__(self, name: str, sense: str = "Minimize") -> None:
        self.name = name
        self.sense = sense
        self.variables: Dict[str, int] = {}
        self.var_lower_bounds: List[float] = []
        self.objective: Dict[int, float] = {}
        self.constraints: List[Tuple[Dict[int, float], str, float]] = []

    def add_variable(self, var_name: str, low_bound: float = 0.0) -> int:
        if var_name in self.variables:
            return self.variables[var_name]
        idx = len(self.variables)
        self.variables[var_name] = idx
        self.var_lower_bounds.append(low_bound)
        return idx

    def set_objective(self, linear: Dict[int, float]) -> None:
        self.objective = dict(linear)

    def add_constraint(self, linear: Dict[int, float], sense: str, rhs: float) -> None:
        assert sense in ("<=", "==", ">="), "Constraint sense must be one of <=, ==, >="
        if sense == ">=":
            linear = {i: -v for i, v in linear.items()}
            rhs = -rhs
            sense = "<="
        self.constraints.append((dict(linear), sense, rhs))

    def to_standard_form(self) -> LPStandardForm:
        num_vars = len(self.variables)
        A: List[List[float]] = []
        b: List[float] = []

        var_names = [None] * num_vars
        for name, idx in self.variables.items():
            var_names[idx] = name

        slack_count = sum(1 for _, sense, _ in self.constraints if sense == "<=")
        total_vars = num_vars + slack_count

        slack_added = 0
        for coefs, sense, rhs in self.constraints:
            row = [0.0] * total_vars
            for j, v in coefs.items():
                row[j] = v
            if sense == "<=":
                slack_col = num_vars + slack_added
                row[slack_col] = 1.0
                slack_added += 1
            A.append(row)
            b.append(rhs)

        c = [0.0] * total_vars
        for j, v in self.objective.items():
            c[j] = v

        if self.sense.lower().startswith("max"):
            c = [-v for v in c]

        var_names_out = var_names + [f"slack_{i}" for i in range(slack_count)]
        return LPStandardForm(A_eq=A, b_eq=b, c=c, var_names=var_names_out)

    def solve(self, method: str = "simplex", **kwargs):
        from .solvers import solve
        lp = self.to_standard_form()
        x, obj = solve(lp, method, **kwargs)
        solution: Dict[str, float] = {}
        for idx, name in enumerate(lp.var_names):
            if name.startswith("slack_"):
                continue
            val = x[idx] if idx < len(x) else 0.0
            solution[name] = val
        true_obj = obj
        if self.sense.lower().startswith("max"):
            true_obj = -obj
        return solution, true_obj, "Optimal"


def LpVariable_dicts(prefix: str, keys: Iterable[Tuple], lowBound: float = 0.0):
    return {k: f"{prefix}_{'_'.join(map(str, k))}" for k in keys}


# ========== Multi-Commodity Flow Problem 数据模型 ==========

@dataclass(frozen=True)
class McfNetwork:
    """
    多商品流网络拓扑表示（与 simplex/dual 的 Network 一致）。

    Attributes:
        nodes: 所有节点 ID 的列表
        adjacency: 每个节点到其邻居列表的映射
        arcs: 所有有向边（节点对元组）的列表
    """
    nodes: List[str]
    adjacency: Dict[str, List[str]]
    arcs: List[Tuple[str, str]] = None

    def __post_init__(self):
        """从邻接表自动构建弧列表。"""
        if self.arcs is None:
            arcs = []
            for u in self.adjacency:
                for v in self.adjacency[u]:
                    arcs.append((u, v))
            # frozen dataclass 需用 object.__setattr__
            object.__setattr__(self, 'arcs', arcs)

    def __repr__(self) -> str:
        return f"McfNetwork(nodes={len(self.nodes)}, arcs={len(self.arcs)})"


@dataclass(frozen=True)
class McfDemand:
    """
    单个商品需求（与 simplex/dual 的 Demand 一致）。

    Attributes:
        source: 源节点 ID
        target: 目标节点 ID
        bandwidth: 带宽需求（流量）
    """
    source: str
    target: str
    bandwidth: float

    def __repr__(self) -> str:
        return f"McfDemand({self.source} → {self.target}, bw={self.bandwidth})"


@dataclass
class McfProblem:
    """
    多商品流问题：网络 + 容量 + 需求列表。

    Attributes:
        network: 网络拓扑
        capacity: 统一容量（当edge_capacities为None时使用）
        demands: 需求列表
        edge_capacities: 每条边的容量字典 {(src, tgt): capacity}，优先于capacity
    """
    network: McfNetwork
    capacity: float = None
    demands: List[McfDemand] = None
    edge_capacities: Dict[Tuple[str, str], float] = None

    def __post_init__(self):
        """验证参数"""
        if self.demands is None:
            object.__setattr__(self, 'demands', [])
        
        # 如果没有指定edge_capacities，且capacity也为None，报错
        if self.edge_capacities is None and self.capacity is None:
            raise ValueError("Must provide either 'capacity' or 'edge_capacities'")

    def get_arc_capacity(self, arc: Tuple[str, str]) -> float:
        """获取指定边的容量"""
        if self.edge_capacities is not None:
            return self.edge_capacities.get(arc, self.capacity or float('inf'))
        return self.capacity or float('inf')

    def __repr__(self) -> str:
        cap_info = f"uniform={self.capacity}" if self.edge_capacities is None else "per-edge"
        return (f"McfProblem(nodes={len(self.network.nodes)}, "
                f"arcs={len(self.network.arcs)}, "
                f"demands={len(self.demands)}, capacity={cap_info})")


def mcf_to_lp(mcf: McfProblem, edge_costs: Dict[Tuple[str, str], float] = None) -> LpProblem:
    """
    将多商品流问题转换为 LpProblem（边基础变量形式）。

    数学模型（与 simplex_solver 一致）:
        变量: f_d(i,j) 表示需求 d 在边 (i,j) 上的流量

        约束:
          1. 流守恒：对每个需求 d 和每个节点 i：
             sum_j f_d(i,j) - sum_j f_d(j,i) =
                 { bw_d    if i = source_d
                 {-bw_d    if i = target_d
                 { 0       otherwise
          2. 容量约束：对每条边 (i,j)：
             sum_d f_d(i,j) <= capacity

        目标: 最小化 sum_{d,e} f_d(e)  (总跳数)

    Args:
        mcf: 多商品流问题实例

    Returns:
        构建好的 LpProblem 对象
    """
    network = mcf.network
    capacity = mcf.capacity
    demands = mcf.demands

    prob = LpProblem(name="MultiCommodityFlow", sense="Minimize")

    # 创建流变量: flow_vars[d_id][(u, v)] -> 变量索引
    flow_vars: Dict[int, Dict[Tuple[str, str], int]] = {}
    for d_id, demand in enumerate(demands):
        flow_vars[d_id] = {}
        for arc in network.arcs:
            var_name = f"f_D{d_id}_{arc[0]}_{arc[1]}"
            var_idx = prob.add_variable(var_name, low_bound=0.0)
            flow_vars[d_id][arc] = var_idx

    # 目标函数：最小化总成本
    obj_dict: Dict[int, float] = {}
    for d_id in range(len(demands)):
        for arc in network.arcs:
            var_idx = flow_vars[d_id][arc]
            # 使用边的成本，如果没有提供则默认为1
            cost = edge_costs.get(arc, 1.0) if edge_costs else 1.0
            obj_dict[var_idx] = obj_dict.get(var_idx, 0.0) + cost
    prob.set_objective(obj_dict)

    # 流守恒约束
    for d_id, demand in enumerate(demands):
        for node in network.nodes:
            # 出流
            out_vars = []
            for neighbor in network.adjacency.get(node, []):
                if (node, neighbor) in flow_vars[d_id]:
                    out_vars.append(flow_vars[d_id][(node, neighbor)])
            # 入流
            in_vars = []
            for prev_node in network.nodes:
                if (prev_node, node) in network.arcs and (prev_node, node) in flow_vars[d_id]:
                    in_vars.append(flow_vars[d_id][(prev_node, node)])

            # 构建约束：outgoing - incoming = rhs
            constraint_coefs: Dict[int, float] = {}
            for v_idx in out_vars:
                constraint_coefs[v_idx] = constraint_coefs.get(v_idx, 0.0) + 1.0
            for v_idx in in_vars:
                constraint_coefs[v_idx] = constraint_coefs.get(v_idx, 0.0) - 1.0

            # 右侧
            if node == demand.source:
                rhs = demand.bandwidth
            elif node == demand.target:
                rhs = -demand.bandwidth
            else:
                rhs = 0.0

            prob.add_constraint(constraint_coefs, "==", rhs)

    # 容量约束（使用每条边各自的容量）
    for arc in network.arcs:
        cap_coefs: Dict[int, float] = {}
        for d_id in range(len(demands)):
            if arc in flow_vars[d_id]:
                var_idx = flow_vars[d_id][arc]
                cap_coefs[var_idx] = cap_coefs.get(var_idx, 0.0) + 1.0
        
        # 获取该边的容量
        arc_capacity = mcf.get_arc_capacity(arc)
        prob.add_constraint(cap_coefs, "<=", arc_capacity)

    return prob


