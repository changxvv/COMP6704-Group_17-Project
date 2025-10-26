"""
Data models for Multi-Commodity Network Flow Column Generation.

This module contains the core data structures: Node, Edge, Demand, and Network.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import logging

from .parser import parse_sndlib_file
from .lmcf_parser import parse_lmcf_files

logger = logging.getLogger(__name__)


class Node:
    """
    Represents a node in the network graph.

    Attributes:
        id: Unique node identifier
        x: X coordinate (optional)
        y: Y coordinate (optional)
        succ: Dictionary mapping successor node IDs to edge IDs
        pred: Dictionary mapping predecessor node IDs to edge IDs
        label: Label for shortest path algorithm (distance)
        sp_pred: Predecessor node in shortest path tree
    """

    def __init__(self, node_id: Any, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """
        Initialize a Node.

        Args:
            node_id: Unique identifier for the node
            x: X coordinate (optional)
            y: Y coordinate (optional)
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.succ: Dict[Any, Any] = {}  # successor_id -> edge_id
        self.pred: Dict[Any, Any] = {}  # predecessor_id -> edge_id
        self.label: float = float('inf')
        self.sp_pred: Optional['Node'] = None

    def add_successor(self, edge_id: Any, succ_id: Any) -> None:
        """Add a successor node with connecting edge."""
        self.succ[succ_id] = edge_id

    def add_predecessor(self, edge_id: Any, pred_id: Any) -> None:
        """Add a predecessor node with connecting edge."""
        self.pred[pred_id] = edge_id

    def __str__(self) -> str:
        return str(self.id)

    def __repr__(self) -> str:
        return f"Node({self.id})"

    def __lt__(self, other: 'Node') -> bool:
        return self.id < other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id


class Edge:
    """
    Represents a directed edge in the network graph.

    Attributes:
        id: Unique edge identifier
        u: Source node
        v: Target node
        capacity: Maximum flow capacity
        weight: Current weight (may change during algorithm)
        init_weight: Initial weight (cost per unit flow)
        capacity_used: Flow currently using this edge
        capacity_left: Remaining capacity
    """

    def __init__(self, edge_id: Any, u: Node, v: Node, weight: float,
                 capacity: float = float('inf')) -> None:
        """
        Initialize an Edge.

        Args:
            edge_id: Unique identifier for the edge
            u: Source node
            v: Target node
            weight: Cost per unit flow
            capacity: Maximum flow capacity (default: infinity)
        """
        self.id = edge_id
        self.u = u
        self.u.add_successor(edge_id, v.id)
        self.v = v
        self.v.add_predecessor(edge_id, u.id)

        self.capacity = capacity
        self.weight = weight
        self.init_weight = weight
        self.capacity_used = 0.0
        self.capacity_left = capacity

    def add_flow(self, amount: float) -> None:
        """
        Update capacity usage.

        Args:
            amount: Amount of capacity to use
        """
        self.capacity_used += amount
        self.capacity_left = self.capacity - self.capacity_used

    def __str__(self) -> str:
        return f"{self.id}: {self.u.id}->{self.v.id} (cost={self.init_weight:.2f}, cap={self.capacity:.2f})"

    def __repr__(self) -> str:
        return f"Edge({self.id}, {self.u.id}->{self.v.id})"


class Demand:
    """
    Represents a commodity demand between origin and destination.

    Attributes:
        id: Unique demand identifier
        origin: Origin node
        destination: Destination node
        quantity: Amount of flow required
        routes: Dictionary of routes (route_hash -> flow_ratio)
    """

    def __init__(self, demand_id: Any, origin: Node, destination: Node, quantity: float) -> None:
        """
        Initialize a Demand.

        Args:
            demand_id: Unique identifier for the demand
            origin: Origin node
            destination: Destination node
            quantity: Amount of flow required
        """
        self.id = demand_id
        self.origin = origin
        self.destination = destination
        self.quantity = quantity
        self.routes: Dict[str, float] = {}

    def hash_route(self, route: list) -> str:
        """
        Create a string hash of a route.

        Args:
            route: List of nodes in the route

        Returns:
            String representation of the route
        """
        return "->".join(str(node.id) for node in route)

    def update_route(self, route: list, ratio: float) -> None:
        """
        Update the routes dictionary with a new route or update existing.

        Args:
            route: List of nodes in the route
            ratio: Proportion of flow on this route
        """
        route_hash = self.hash_route(route)
        self.routes[route_hash] = self.routes.get(route_hash, 0.0) + ratio

    # class Demand 内（确保 __init__ 里有 self.routes = {}）
    def add_route(self, route, ratio: float):
        """将路径写入 routes，并累加占比。允许 route 是 Node 列表或 node_id 列表。"""

        def _nid(n):
            return n.id if hasattr(n, "id") else n

        # 若已有 hash_route() 就用原来的；否则简单拼接 id
        route_hash = self.hash_route(route) if hasattr(self, "hash_route") \
            else "-".join(str(_nid(n)) for n in route)
        self.routes[route_hash] = self.routes.get(route_hash, 0.0) + float(ratio)


    def __str__(self) -> str:
        return f"D{self.id}: {self.origin.id}->{self.destination.id} (q={self.quantity:.2f})"

    def __repr__(self) -> str:
        return f"Demand({self.id}, {self.origin.id}->{self.destination.id})"


class Network:
    """
    Represents the entire network with nodes, edges, and demands.

    Attributes:
        nodes: Dictionary mapping node IDs to Node objects
        edges: Dictionary mapping edge IDs to Edge objects
        demands: Dictionary mapping demand IDs to Demand objects
        edge_dict: Dictionary mapping (source_id, target_id) to edge_id
    """

    def __init__(self) -> None:
        """Initialize an empty Network."""
        self.nodes: Dict[Any, Node] = {}
        self.edges: Dict[Any, Edge] = {}
        self.demands: Dict[Any, Demand] = {}
        self.edge_dict: Dict[tuple, Any] = {}  # (u_id, v_id) -> edge_id


    def add_node(self, node_id: Any, x: Optional[float] = None, y: Optional[float] = None) -> Node:
        """
        Add a node to the network.

        Args:
            node_id: Unique identifier for the node
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            The created Node object
        """
        node = Node(node_id, x, y)
        self.nodes[node_id] = node
        return node

    def add_edge(self, edge_id: Any, u_id: Any, v_id: Any, weight: float,
                 capacity: float = float('inf')) -> Edge:
        """
        Add an edge to the network. Creates nodes if they don't exist.

        Args:
            edge_id: Unique identifier for the edge
            u_id: Source node ID
            v_id: Target node ID
            weight: Cost per unit flow
            capacity: Maximum flow capacity (default: infinity)

        Returns:
            The created Edge object
        """
        if u_id not in self.nodes:
            self.add_node(u_id)
        if v_id not in self.nodes:
            self.add_node(v_id)

        u = self.nodes[u_id]
        v = self.nodes[v_id]
        edge = Edge(edge_id, u, v, weight, capacity)
        self.edges[edge_id] = edge
        self.edge_dict[(u_id, v_id)] = edge_id
        return edge

    def add_demand(self, demand_id: Any, origin_id: Any, dest_id: Any, quantity: float) -> Demand:
        """
        Add a demand to the network.

        Args:
            demand_id: Unique identifier for the demand
            origin_id: Origin node ID
            dest_id: Destination node ID
            quantity: Amount of flow required

        Returns:
            The created Demand object

        Raises:
            ValueError: If origin or destination node doesn't exist
        """
        if origin_id not in self.nodes:
            raise ValueError(f"Origin node {origin_id} not found")
        if dest_id not in self.nodes:
            raise ValueError(f"Destination node {dest_id} not found")

        origin = self.nodes[origin_id]
        destination = self.nodes[dest_id]
        demand = Demand(demand_id, origin, destination, quantity)
        self.demands[demand_id] = demand
        return demand

    def load_from_sndlib(self, filepath: Path, undirected: bool = True) -> None:
        """
        Load network from an SNDlib format file.

        Args:
            filepath: Path to the SNDlib format file
            undirected: If True, treat links as undirected (create reverse edges).
                       Set to False if the input already has bidirectional links.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        logger.info(f"Loading network from {filepath} (undirected={undirected})")
        data = parse_sndlib_file(filepath)

        # Add nodes
        for node_id, node_data in data['nodes'].items():
            self.add_node(node_id, node_data['x'], node_data['y'])

        # Add edges
        for edge_id, edge_data in data['links'].items():
            # Add forward edge
            self.add_edge(
                edge_id,
                edge_data['source'],
                edge_data['target'],
                edge_data['unit_cost'],
                edge_data['capacity']
            )

            # Add reverse edge if undirected
            if undirected:
                reverse_edge_id = f"{edge_id}_reverse"
                self.add_edge(
                    reverse_edge_id,
                    edge_data['target'],
                    edge_data['source'],
                    edge_data['unit_cost'],
                    edge_data['capacity']
                )

        # Add demands
        for demand_id, demand_data in data['demands'].items():
            self.add_demand(
                demand_id,
                demand_data['source'],
                demand_data['target'],
                demand_data['demand_value']
            )

        logger.info(f"Loaded network: {len(self.nodes)} nodes, "
                   f"{len(self.edges)} edges, {len(self.demands)} demands")

    def load_from_lmcf(self, network_file: Path, demand_file: Path, undirected: bool = True) -> None:
        """
        Load network from LMCF format files.

        Args:
            network_file: Path to the network file (C*.txt)
            demand_file: Path to the demand file (D*.txt)
            undirected: If True, treat links as undirected (create reverse edges).
                       Default is True as all LMCF problems are undirected.

        Raises:
            FileNotFoundError: If either file doesn't exist
            ValueError: If the file format is invalid
        """
        logger.info(f"Loading LMCF network from {network_file.name} and {demand_file.name} "
                   f"(undirected={undirected})")
        data = parse_lmcf_files(network_file, demand_file)

        # Add nodes
        for node_id, node_data in data['nodes'].items():
            self.add_node(node_id, node_data['x'], node_data['y'])

        # Add edges
        for edge_id, edge_data in data['links'].items():
            # Add forward edge
            self.add_edge(
                edge_id,
                edge_data['source'],
                edge_data['target'],
                edge_data['unit_cost'],
                edge_data['capacity']
            )

            # Add reverse edge if undirected
            if undirected:
                reverse_edge_id = f"{edge_id}_reverse"
                self.add_edge(
                    reverse_edge_id,
                    edge_data['target'],
                    edge_data['source'],
                    edge_data['unit_cost'],
                    edge_data['capacity']
                )

        # Add demands
        for demand_id, demand_data in data['demands'].items():
            self.add_demand(
                demand_id,
                demand_data['source'],
                demand_data['target'],
                demand_data['demand_value']
            )

        logger.info(f"Loaded network: {len(self.nodes)} nodes, "
                   f"{len(self.edges)} edges, {len(self.demands)} demands")

    def __str__(self) -> str:
        return f"Network(nodes={len(self.nodes)}, edges={len(self.edges)}, demands={len(self.demands)})"

    def __repr__(self) -> str:
        return self.__str__()
