"""
Parsers for network and demand data formats.

Supports:
- SNDlib native format
- LMCF format (tab-separated)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from .data_models import Network, Demand


def parse_sndlib(filepath: Path) -> Tuple[Network, List[Demand], float]:
    """
    Parse SNDlib native format file.

    Args:
        filepath: Path to SNDlib format file

    Returns:
        Tuple of (network, demands, link_capacity)
    """
    adjacency: Dict[str, List[str]] = {}
    demands: List[Demand] = []

    nodes = set()
    arcs = []
    link_capacities = {}
    link_costs = {}

    current_section = None

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse sections using regex
    sections = re.split(r'^([A-Z_]+)\s*\(', content, flags=re.MULTILINE)

    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        section_content = sections[i+1].split(')')[0]

        if section_name == 'NODES':
            current_section = 'NODES'
            for line in section_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'(\w+)\s*\([^)]*\)', line)
                if match:
                    node = match.group(1)
                    nodes.add(node)
                    adjacency.setdefault(node, [])

        elif section_name == 'LINKS':
            current_section = 'LINKS'
            for line in section_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)', line)
                if match:
                    link_id, node1, node2 = match.groups()

                    adjacency.setdefault(node1, [])
                    adjacency.setdefault(node2, [])

                    # Add bidirectional links
                    if node2 not in adjacency[node1]:
                        adjacency[node1].append(node2)
                    if node1 not in adjacency[node2]:
                        adjacency[node2].append(node1)

                    arcs.append((node1, node2))
                    arcs.append((node2, node1))

                    # Extract capacity (best-effort)
                    capacity_match = re.search(r'\(\s*([\d.\s]+)\s*\)', line)
                    if capacity_match:
                        try:
                            capacity_values = [float(x) for x in capacity_match.group(1).split()]
                            if capacity_values:
                                default_capacity = capacity_values[0]
                                link_capacities[(node1, node2)] = default_capacity
                                link_capacities[(node2, node1)] = default_capacity
                        except ValueError:
                            pass

                    # Extract routing cost (best-effort)
                    cost_match = re.search(r'\)\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                    if cost_match:
                        try:
                            routing_cost = float(cost_match.group(3))
                            link_costs[(node1, node2)] = routing_cost
                            link_costs[(node2, node1)] = routing_cost
                        except ValueError:
                            link_costs[(node1, node2)] = 1.0
                            link_costs[(node2, node1)] = 1.0
                    else:
                        link_costs[(node1, node2)] = 1.0
                        link_costs[(node2, node1)] = 1.0

        elif section_name == 'DEMANDS':
            current_section = 'DEMANDS'
            for line in section_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)\s+\d+\s+([\d.]+)', line)
                if match:
                    demand_id, source, target, bandwidth = match.groups()
                    try:
                        demands.append(Demand(source, target, float(bandwidth), demand_id))
                    except ValueError:
                        pass

    # Determine default capacity
    if link_capacities:
        capacity_counter = Counter(link_capacities.values())
        default_capacity = capacity_counter.most_common(1)[0][0]
    else:
        default_capacity = 1000000.0

    # Create network object
    network = Network(nodes=list(nodes), adjacency=adjacency, arcs=arcs)
    network.link_capacities = link_capacities
    network.link_costs = link_costs

    return network, demands, default_capacity


def parse_lmcf_network(filepath: Path) -> Tuple[Network, float]:
    """
    Parse LMCF network file format.

    Format: source target capacity cost (tab separated)
    Example: 1	2	91	139
    """
    adjacency: Dict[str, List[str]] = {}
    arcs = []
    link_capacities = {}
    link_costs = {}
    nodes = set()

    default_capacity = 0.0

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    source = parts[0]
                    target = parts[1]
                    cost = float(parts[2])      # 第3个字段是cost
                    capacity = float(parts[3])  # 第4个字段是capacity

                    nodes.add(source)
                    nodes.add(target)

                    adjacency.setdefault(source, [])
                    adjacency.setdefault(target, [])

                    if target not in adjacency[source]:
                        adjacency[source].append(target)
                    if source not in adjacency[target]:
                        adjacency[target].append(source)

                    arcs.append((source, target))

                    link_capacities[(source, target)] = capacity
                    link_capacities[(target, source)] = capacity
                    link_costs[(source, target)] = cost
                    link_costs[(target, source)] = cost

                    if capacity > default_capacity:
                        default_capacity = capacity

                except ValueError:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    continue

    network = Network(
        nodes=list(nodes),
        adjacency=adjacency,
        arcs=arcs
    )
    network.link_capacities = link_capacities
    network.link_costs = link_costs

    return network, default_capacity


def parse_lmcf_demands(filepath: Path) -> List[Demand]:
    """
    Parse LMCF demand file format.

    Format: source target bandwidth (tab separated)
    Example: 1	2	100.0
    """
    demands = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    source = parts[0]
                    target = parts[1]
                    bandwidth = float(parts[2])

                    demands.append(Demand(source, target, bandwidth))

                except ValueError:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    continue

    return demands


def parse_lmcf_files(network_file: Path, demand_file: Path, undirected: bool = False) -> Tuple[Network, List[Demand], float]:
    """
    Parse both LMCF network and demand files.

    Args:
        network_file: Path to network file
        demand_file: Path to demand file
        undirected: Whether the network is undirected

    Returns:
        Tuple of (network, demands, default_capacity)
    """
    network, default_capacity = parse_lmcf_network(network_file)
    demands = parse_lmcf_demands(demand_file)

    # Ensure attributes exist on network
    if not hasattr(network, 'link_capacities'):
        network.link_capacities = {}
    if not hasattr(network, 'link_costs'):
        network.link_costs = {}

    print(f"Parsed LMCF: {len(network.nodes)} nodes, {len(network.arcs)} arcs, {len(demands)} demands")

    return network, demands, default_capacity
