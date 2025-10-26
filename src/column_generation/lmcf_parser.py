"""
LMCF Format Parser for Multi-Commodity Network Flow.

This module provides functionality to parse LMCF (Linear Multi-Commodity Flow)
format files containing network topology and demand information.

LMCF format consists of two separate files:
- Network file (C*.txt): startnode endnode cost capacity
- Demand file (D*.txt): origin destination demand
"""

import re
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def parse_lmcf_files(network_file: Path, demand_file: Path) -> Dict[str, Any]:
    """
    Parse LMCF format files and extract nodes, links, and demands.

    Args:
        network_file: Path to the network file (C*.txt)
        demand_file: Path to the demand file (D*.txt)

    Returns:
        Dictionary containing 'nodes', 'links', and 'demands' data
        in the same format as parse_sndlib_file()

    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If the file format is invalid
    """
    network_file = Path(network_file)
    demand_file = Path(demand_file)

    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")
    if not demand_file.exists():
        raise FileNotFoundError(f"Demand file not found: {demand_file}")

    logger.info(f"Parsing LMCF files: {network_file.name}, {demand_file.name}")

    # Parse network file
    links, nodes_from_links = _parse_network_file(network_file)

    # Parse demand file
    demands, nodes_from_demands = _parse_demand_file(demand_file)

    # Combine all nodes (from both links and demands)
    all_node_ids = nodes_from_links | nodes_from_demands
    nodes = {node_id: {'x': None, 'y': None} for node_id in all_node_ids}

    logger.info(f"Parsed {len(nodes)} nodes, {len(links)} links, {len(demands)} demands")

    return {
        'nodes': nodes,
        'links': links,
        'demands': demands
    }


def _parse_network_file(filepath: Path) -> tuple[Dict[str, Dict[str, Any]], set]:
    """
    Parse the network file (C*.txt) to extract links.

    Format: startnode endnode cost capacity (space/tab separated)

    Args:
        filepath: Path to the network file

    Returns:
        Tuple of (links_dict, node_ids_set)
    """
    links = {}
    node_ids = set()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse line: startnode endnode cost capacity
            parts = line.split()

            if len(parts) < 4:
                logger.warning(f"Skipping invalid line {line_num} in {filepath.name}: {line}")
                continue

            try:
                start_node = parts[0]
                end_node = parts[1]
                cost = float(parts[2])
                capacity = float(parts[3])

                # Convert node IDs to strings for consistency
                start_node_str = str(start_node)
                end_node_str = str(end_node)

                # Track nodes
                node_ids.add(start_node_str)
                node_ids.add(end_node_str)

                # Create link ID
                link_id = f"L{start_node_str}_{end_node_str}"

                links[link_id] = {
                    'source': start_node_str,
                    'target': end_node_str,
                    'capacity': capacity,
                    'unit_cost': cost,
                    'routing_cost': cost,
                    'capacity_modules': []  # LMCF format has no modules
                }

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num} in {filepath.name}: {e}")
                continue

    return links, node_ids


def _parse_demand_file(filepath: Path) -> tuple[Dict[str, Dict[str, Any]], set]:
    """
    Parse the demand file (D*.txt) to extract demands.

    Format: origin destination demand (space/tab separated)

    Args:
        filepath: Path to the demand file

    Returns:
        Tuple of (demands_dict, node_ids_set)
    """
    demands = {}
    node_ids = set()
    demand_counter = {}  # Track multiple demands between same OD pair

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse line: origin destination demand
            parts = line.split()

            if len(parts) < 3:
                logger.warning(f"Skipping invalid line {line_num} in {filepath.name}: {line}")
                continue

            try:
                origin = parts[0]
                destination = parts[1]
                demand_value = float(parts[2])

                # Convert node IDs to strings for consistency
                origin_str = str(origin)
                destination_str = str(destination)

                # Track nodes
                node_ids.add(origin_str)
                node_ids.add(destination_str)

                # Create unique demand ID
                # If multiple demands between same OD pair, add index
                od_pair = (origin_str, destination_str)
                if od_pair in demand_counter:
                    demand_counter[od_pair] += 1
                    demand_id = f"D{origin_str}_{destination_str}_{demand_counter[od_pair]}"
                else:
                    demand_counter[od_pair] = 0
                    demand_id = f"D{origin_str}_{destination_str}"

                demands[demand_id] = {
                    'source': origin_str,
                    'target': destination_str,
                    'routing_unit': 1,
                    'demand_value': demand_value,
                    'max_path_length': 9999  # No limit specified in LMCF format
                }

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num} in {filepath.name}: {e}")
                continue

    return demands, node_ids
