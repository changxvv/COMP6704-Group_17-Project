"""
Pricing Subproblem Solver for Column Generation.

This module implements the pricing subproblem using Dijkstra's algorithm
to find minimum reduced cost paths.
"""

import heapq
from typing import Dict, List, Tuple, Any, Optional
import logging

from .data_models import Network, Node, Edge
from .utils import reset_node_labels, reset_edge_capacity, reset_edge_weight

logger = logging.getLogger(__name__)


def dijkstra(network: Network, source: Node, destination: Node) -> Tuple[List[Node], float]:
    """
    Find shortest path from source to destination using Dijkstra's algorithm.

    Uses current edge weights which may be modified by dual variables.

    Args:
        network: The network graph
        source: Source node
        destination: Destination node

    Returns:
        Tuple of (path as list of nodes, shortest distance)
    """
    # Reset labels
    reset_node_labels(network)

    # Initialize source
    source.label = 0.0
    source.sp_pred = None

    # Min heap: (label, node)
    min_heap = [(0.0, source)]

    while min_heap:
        current_label, current_node = heapq.heappop(min_heap)

        # If we reached destination, reconstruct path
        if current_node.id == destination.id:
            path = [destination]
            pred = destination.sp_pred
            while pred is not None:
                path.append(pred)
                pred = pred.sp_pred
            path.reverse()
            return path, destination.label

        # Skip if we've already found a better path to this node
        if current_label > current_node.label:
            continue

        # Explore successors
        for succ_id, edge_id in current_node.succ.items():
            succ_node = network.nodes[succ_id]
            edge = network.edges[edge_id]

            # Calculate new label
            new_label = current_label + edge.weight

            # Update if better path found
            if new_label < succ_node.label:
                succ_node.label = new_label
                succ_node.sp_pred = current_node
                heapq.heappush(min_heap, (new_label, succ_node))

    # No path found
    logger.warning(f"No path found from {source.id} to {destination.id}")
    return [], float('inf')


def generate_initial_solution(network: Network) -> Dict[str, Any]:
    """
    Generate an initial feasible solution by assigning all flow to shortest paths.

    Args:
        network: The network

    Returns:
        Solution dictionary containing routes, flows, and cost
    """
    logger.info("Generating initial solution")

    # Reset edge capacities and weights
    reset_edge_capacity(network)
    reset_edge_weight(network)

    solution = {
        'routes': {},
        'flow': {},
        'cost': 0.0,
        'reducedCost': 0.0
    }

    # For each demand, find shortest path and assign flow
    for demand_id, demand in network.demands.items():
        path, distance = dijkstra(network, demand.origin, demand.destination)

        if not path:
            logger.error(f"No path found for demand {demand_id}")
            continue

        # Store route
        solution['routes'][demand_id] = path

        # Update edge flows and compute cost
        for i in range(len(path) - 1):
            u_id = path[i].id
            v_id = path[i + 1].id
            edge_id = network.edge_dict[(u_id, v_id)]
            edge = network.edges[edge_id]

            # Update capacity
            edge.use_capacity(demand.quantity)

            # Add to cost
            solution['cost'] += demand.quantity * edge.init_weight
            solution['reducedCost'] += demand.quantity * edge.weight

    # Calculate flows on each edge
    for edge_id, edge in network.edges.items():
        solution['flow'][edge_id] = edge.capacity_used

    logger.info(f"Initial solution cost: {solution['cost']:.2f}")
    return solution


def solve_pricing_problem(
    network: Network,
    dual_vars: List[float],
    bounded_edges: Dict[Any, Edge]
) -> Tuple[Dict[str, Any], float]:
    """
    Solve the pricing subproblem to find a new column with negative reduced cost.

    The pricing problem finds paths for each commodity using modified edge weights:
    w'[e] = w[e] - π[e]

    Args:
        network: The network
        dual_vars: List of dual variables (π for capacity constraints, μ for convexity)
        bounded_edges: Dictionary of edges with capacity constraints

    Returns:
        Tuple of (solution dictionary, reduced cost)
    """
    logger.debug("Solving pricing subproblem")

    # Adjust edge weights using dual variables
    # dual_vars = [π_1, π_2, ..., π_m, μ] where m is number of bounded edges
    edge_list = list(bounded_edges.keys())
    for i, edge_id in enumerate(edge_list):
        edge = network.edges[edge_id]
        edge.weight = edge.init_weight - dual_vars[i]

    # Reset edge capacities
    reset_edge_capacity(network)

    # Generate new solution (extreme point)
    solution = {
        'routes': {},
        'flow': {},
        'cost': 0.0,
        'reducedCost': 0.0
    }

    # For each demand, find shortest path with modified weights
    for demand_id, demand in network.demands.items():
        path, distance = dijkstra(network, demand.origin, demand.destination)

        if not path:
            logger.warning(f"No path found for demand {demand_id} in pricing problem")
            continue

        # Store route
        solution['routes'][demand_id] = path

        # Update edge flows and compute cost
        for i in range(len(path) - 1):
            u_id = path[i].id
            v_id = path[i + 1].id
            edge_id = network.edge_dict[(u_id, v_id)]
            edge = network.edges[edge_id]

            # Update capacity
            edge.use_capacity(demand.quantity)

            # Add to original cost
            solution['cost'] += demand.quantity * edge.init_weight
            # Add to reduced cost (using modified weights)
            solution['reducedCost'] += demand.quantity * edge.weight

    # Calculate flows on each edge
    for edge_id, edge in network.edges.items():
        solution['flow'][edge_id] = edge.capacity_used

    # Calculate final reduced cost: c̄ = c - Σ(π * a) - μ
    # The reducedCost field already has Σ(w' * demand) = Σ((w - π) * demand)
    # We need to compute: -reducedCost + μ
    mu = dual_vars[-1]  # Last dual variable is for convexity constraint
    reduced_cost = -solution['reducedCost'] + mu

    logger.debug(f"Pricing solution cost: {solution['cost']:.2f}, reduced cost: {reduced_cost:.6f}")

    # Reset edge weights
    reset_edge_weight(network)

    return solution, reduced_cost
