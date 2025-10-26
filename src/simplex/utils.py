"""
Enhanced utility functions for advanced simplex methods.
"""

import heapq
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_models import Network, Demand, Path, Solution


def validate_basis(A: np.ndarray, basis_indices: np.ndarray, tolerance: float = 1e-8) -> bool:
    """
    Validate that basis matrix is non-singular.

    Args:
        A: Constraint matrix
        basis_indices: Indices of basic variables
        tolerance: Numerical tolerance

    Returns:
        True if basis is valid
    """
    try:
        B = A[:, basis_indices]
        # Check condition number
        cond = np.linalg.cond(B)
        return cond < 1 / tolerance
    except Exception:
        return False


def compute_reduced_costs(c: np.ndarray, A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute reduced costs for given dual variables.

    Args:
        c: Objective coefficients
        A: Constraint matrix
        y: Dual variables

    Returns:
        Reduced costs vector
    """
    return c - A.T @ y


def compute_step_size(x: np.ndarray, d: np.ndarray, tolerance: float = 1e-8) -> float:
    """
    Compute maximum step size maintaining feasibility.

    Args:
        x: Current solution
        d: Direction vector
        tolerance: Numerical tolerance

    Returns:
        Maximum feasible step size
    """
    # Find maximum step such that x + αd ≥ 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = -x / d
        ratios[d >= -tolerance] = np.inf  # Ignore non-decreasing directions
        return np.min(ratios[ratios > 0]) if np.any(ratios > 0) else 0.0


def is_optimal(x: np.ndarray, r: np.ndarray, tolerance: float = 1e-8) -> bool:
    """
    Check optimality conditions.

    Args:
        x: Primal solution
        r: Reduced costs
        tolerance: Numerical tolerance

    Returns:
        True if optimality conditions are satisfied
    """
    # Primal feasibility: x ≥ 0 (already assumed)
    # Dual feasibility: r ≥ 0
    # Complementary slackness: xᵢrᵢ = 0
    return (np.all(r >= -tolerance) and
            np.allclose(x * r, 0, atol=tolerance))


def pivot_rule_bland(A: np.ndarray, r: np.ndarray, tolerance: float = 1e-8) -> int:
    """
    Bland's rule for pivot selection.

    Args:
        A: Constraint matrix (not used in basic Bland's rule)
        r: Reduced costs
        tolerance: Numerical tolerance

    Returns:
        Index of entering variable
    """
    # Select first variable with negative reduced cost
    negative_indices = np.where(r < -tolerance)[0]
    return int(negative_indices[0]) if len(negative_indices) > 0 else -1


def pivot_rule_dantzig(r: np.ndarray, tolerance: float = 1e-8) -> int:
    """
    Dantzig's rule for pivot selection.

    Args:
        r: Reduced costs
        tolerance: Numerical tolerance

    Returns:
        Index of entering variable
    """
    # Select variable with most negative reduced cost
    negative_indices = np.where(r < -tolerance)[0]
    if len(negative_indices) > 0:
        return int(negative_indices[np.argmin(r[negative_indices])])
    return -1


def dijkstra_shortest_path(
    network: Network,
    source: str,
    target: str,
    costs: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[List[Tuple[str, str]], float]:
    """
    Find shortest path using Dijkstra's algorithm.
    """
    if costs is None and hasattr(network, 'link_costs'):
        costs = network.link_costs
    elif costs is None:
        costs = {arc: 1.0 for arc in network.arcs}

    pq = [(0.0, source)]
    distances = {node: float('inf') for node in network.nodes}
    distances[source] = 0.0
    parent: Dict[str, Tuple[str, str]] = {}

    while pq:
        dist, u = heapq.heappop(pq)

        if dist > distances[u]:
            continue

        if u == target:
            break

        for v in network.adjacency.get(u, []):
            edge = (u, v)
            edge_cost = costs.get(edge, 1.0)
            new_dist = dist + edge_cost

            if new_dist < distances[v]:
                distances[v] = new_dist
                parent[v] = (u, edge)
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    if target not in parent and target != source:
        return [], float('inf')

    path_edges = []
    current = target
    while current != source:
        if current not in parent:
            return [], float('inf')
        parent_node, edge = parent[current]
        path_edges.append(edge)
        current = parent_node

    path_edges.reverse()
    return path_edges, distances[target]


def validate_solution(
    solution: Solution,
    network: Network,
    demands: List[Demand],
    capacity: float
) -> bool:
    """
    Validate solution for MCNF problem.
    """
    if solution.status != "optimal":
        print(f"Cannot validate non-optimal solution (status: {solution.status})")
        return False

    # Check flow conservation: in our path representation we expect per-demand weights sum to 1
    for demand_id, paths_weights in solution.allocation.items():
        if demand_id >= len(demands):
            print(f"Invalid demand ID: {demand_id}")
            return False

        total_weight = sum(weight for _, weight in paths_weights)
        if abs(total_weight - 1.0) > 1e-4:
            print(f"Flow conservation violated for demand {demand_id}: total weight = {total_weight}")
            return False

    # Check capacity constraints
    link_usage: Dict[Tuple[str, str], float] = {}

    for demand_id, paths_weights in solution.allocation.items():
        demand = demands[demand_id]
        for path, weight in paths_weights:
            flow = demand.bandwidth * weight
            for edge in path.edges:
                link_usage[edge] = link_usage.get(edge, 0.0) + flow

    for arc in network.arcs:
        usage = link_usage.get(arc, 0.0)
        arc_capacity = getattr(network, 'link_capacities', {}).get(arc, capacity)

        if usage > arc_capacity + 1e-4:
            print(f"Capacity constraint violated on arc {arc}: usage = {usage:.2f}, capacity = {arc_capacity:.2f}")
            return False

    return True


def compute_objective(
    allocation: Dict[int, List[Tuple[Path, float]]],
    demands: List[Demand],
    network: Optional[Network] = None
) -> float:
    """
    Compute objective value.
    """
    total_cost = 0.0

    for demand_id, paths_weights in allocation.items():
        if demand_id >= len(demands):
            continue

        demand = demands[demand_id]
        for path, weight in paths_weights:
            if hasattr(path, 'cost') and path.cost > 0:
                path_cost = path.cost
            else:
                path_cost = len(path.edges)
                if network and hasattr(network, 'link_costs'):
                    path_cost = sum(network.link_costs.get(edge, 1.0) for edge in path.edges)

            total_cost += path_cost * demand.bandwidth * weight

    return total_cost


def find_flow_path(
    flows: Dict[Tuple[str, str], float],
    source: str,
    target: str
) -> Tuple[List[str], float]:
    """
    Find a path with positive flow using BFS.
    """
    queue = [(source, [source], float('inf'))]
    visited = set()

    while queue:
        current, path, min_flow = queue.pop(0)

        if current == target:
            return path, min_flow

        if current in visited:
            continue
        visited.add(current)

        for arc, flow in flows.items():
            if arc[0] == current and flow > 1e-6 and arc[1] not in path:
                new_min_flow = min(min_flow, flow)
                queue.append((arc[1], path + [arc[1]], new_min_flow))

    return None, 0.0
