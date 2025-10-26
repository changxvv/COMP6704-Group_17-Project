"""
Enhanced data models for advanced simplex methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class LinearProgram:
    """
    Standard form linear program representation.

    min cᵀx
    s.t. Ax = b
         x ≥ 0
    """
    c: np.ndarray  # Objective coefficients
    A: np.ndarray  # Constraint matrix
    b: np.ndarray  # Right-hand side
    variable_names: List[str] = field(default_factory=list)
    constraint_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate dimensions."""
        m, n = self.A.shape
        assert len(self.c) == n, "c must have same dimension as columns of A"
        assert len(self.b) == m, "b must have same dimension as rows of A"


@dataclass
class Basis:
    """
    LP basis representation for advanced simplex methods.
    """
    basic_indices: np.ndarray  # Indices of basic variables
    nonbasic_indices: np.ndarray  # Indices of nonbasic variables
    B_inv: Optional[np.ndarray] = None  # Inverse of basis matrix
    primal_solution: Optional[np.ndarray] = None
    dual_solution: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None


@dataclass
class PrimalDualState:
    """
    State for primal-dual simplex algorithm.
    """
    primal: np.ndarray  # Primal variables
    dual: np.ndarray    # Dual variables
    slack: np.ndarray   # Slack variables
    working_set: np.ndarray  # Working set indices
    iteration: int = 0
    objective: float = 0.0


@dataclass
class Network:
    """
    Network topology representation.
    """
    nodes: List[str]
    adjacency: Dict[str, List[str]]
    arcs: List[Tuple[str, str]] = field(default_factory=list)
    link_capacities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    link_costs: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self):
        """Build arc list from adjacency structure if not provided."""
        if not self.arcs:
            arcs = []
            for u in self.adjacency:
                for v in self.adjacency[u]:
                    arcs.append((u, v))
            object.__setattr__(self, 'arcs', arcs)

    def __repr__(self) -> str:
        return f"Network(nodes={len(self.nodes)}, arcs={len(self.arcs)})"


@dataclass(frozen=True)
class Demand:
    """
    A single commodity demand.
    """
    source: str
    target: str
    bandwidth: float
    demand_id: Optional[str] = None

    def __repr__(self) -> str:
        id_str = f" {self.demand_id}" if self.demand_id else ""
        return f"Demand{id_str}({self.source} → {self.target}, bw={self.bandwidth})"


@dataclass(frozen=True)
class Path:
    """
    A path through the network.
    """
    edges: Tuple[Tuple[str, str], ...]
    cost: float
    demand_id: Optional[int] = None

    def __repr__(self) -> str:
        if not self.edges:
            return "Path(empty)"
        nodes = [self.edges[0][0]] + [e[1] for e in self.edges]
        path_str = " → ".join(nodes)
        demand_str = f" for D{self.demand_id}" if self.demand_id is not None else ""
        return f"Path({path_str}, cost={self.cost:.2f}{demand_str})"


@dataclass
class Solution:
    """
    Final solution from optimization.
    """
    allocation: Dict[int, List[Tuple[Path, float]]]
    objective: float
    iterations: int
    time: float
    converged: bool = True
    status: str = "optimal"
    algorithm: str = "simplex"

    # 额外统计字段（可选）
    primal_residual: Optional[float] = None
    dual_residual: Optional[float] = None
    complementarity: Optional[float] = None
    lp_var_count: Optional[int] = None
    termination_reason: Optional[str] = None

    def __repr__(self) -> str:
        num_demands = len(self.allocation)
        total_paths = sum(len(paths) for paths in self.allocation.values())
        avg_paths = total_paths / num_demands if num_demands > 0 else 0
        return (f"Solution(algorithm={self.algorithm}, status={self.status}, "
                f"obj={self.objective:.2f}, iters={self.iterations}, "
                f"time={self.time:.2f}s, avg_paths={avg_paths:.2f})")
