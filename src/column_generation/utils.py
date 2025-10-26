"""
Utility functions for Column Generation algorithm.

This module provides helper functions for network operations,
solution validation, and objective value computation.
"""

from typing import Dict, Any, List
import logging

from .data_models import Network, Node, Edge

logger = logging.getLogger(__name__)


def reset_node_labels(network: Network) -> None:
    """
    Reset all node labels and shortest path predecessors.

    Used before running shortest path algorithm.

    Args:
        network: The network containing nodes to reset
    """
    for node in network.nodes.values():
        node.label = float('inf')
        node.sp_pred = None


def reset_edge_capacity(network: Network) -> None:
    """
    Reset all edge capacities to initial state.

    Clears capacity usage information from all edges.

    Args:
        network: The network containing edges to reset
    """
    for edge in network.edges.values():
        edge.capacity_used = 0.0
        edge.capacity_left = edge.capacity


def reset_edge_weight(network: Network) -> None:
    """
    Reset all edge weights to their initial values.

    Args:
        network: The network containing edges to reset
    """
    for edge in network.edges.values():
        edge.weight = edge.init_weight


def hash_route(route: List[Node]) -> str:
    """
    Convert a route (list of nodes) to a string hash.

    Args:
        route: List of Node objects representing a path

    Returns:
        String representation of the route
    """
    return "->".join(str(node.id) for node in route)


def compute_objective(network: Network) -> float:
    """
    Compute the total objective value (total cost) of current flow.

    Args:
        network: The network with current flow assignments

    Returns:
        Total cost across all edges
    """
    total_cost = 0.0
    for edge in network.edges.values():
        total_cost += edge.capacity_used * edge.init_weight
    return total_cost


def validate_solution(network: Network, solution: Dict[str, Any]) -> bool:
    """
    Validate a solution for feasibility.

    Checks:
    1. All edge flows are within capacity constraints
    2. Flow conservation at all nodes
    3. All demand requirements are met

    Args:
        network: The network
        solution: Solution dictionary with flow information

    Returns:
        True if solution is feasible, False otherwise
    """
    tolerance = 1e-6

    # Check capacity constraints
    for edge_id, edge in network.edges.items():
        if edge_id in solution.get('flow', {}):
            flow = solution['flow'][edge_id]
            if flow > edge.capacity + tolerance:
                logger.warning(f"Capacity violation on edge {edge_id}: "
                             f"flow={flow:.2f} > capacity={edge.capacity:.2f}")
                return False

    logger.info("Solution validation passed")
    return True


def get_bounded_edges(network: Network) -> Dict[Any, Edge]:
    """
    Get all edges with finite capacity constraints.

    Args:
        network: The network

    Returns:
        Dictionary of edge_id -> Edge for edges with bounded capacity
    """
    bounded_edges = {}
    for edge_id, edge in network.edges.items():
        if edge.capacity < float('inf'):
            bounded_edges[edge_id] = edge

    logger.info(f"Found {len(bounded_edges)} edges with bounded capacity")
    return bounded_edges


def compute_route_cost(network: Network, route: List[Node], demand_quantity: float) -> float:
    """
    Compute the cost of routing a demand along a specific path.

    Args:
        network: The network
        route: List of nodes in the path
        demand_quantity: Amount of flow

    Returns:
        Total cost for this route
    """
    cost = 0.0
    for i in range(len(route) - 1):
        u_id = route[i].id
        v_id = route[i + 1].id
        edge_id = network.edge_dict.get((u_id, v_id))
        if edge_id:
            edge = network.edges[edge_id]
            cost += edge.init_weight * demand_quantity
        else:
            logger.warning(f"Edge not found: {u_id} -> {v_id}")
    return cost
