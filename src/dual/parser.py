"""
SNDlib Format Parser for Multi-Commodity Network Flow.

This module provides functionality to parse SNDlib native format files
containing network topology and demand information.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parse_sndlib_file(filepath: Path) -> Dict[str, Any]:
    """
    Parse an SNDlib format file and extract nodes, links, and demands.

    Args:
        filepath: Path to the SNDlib format file

    Returns:
        Dictionary containing 'nodes', 'links', and 'demands' data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    logger.info(f"Parsing SNDlib file: {filepath}")

    # Parse each section
    nodes = _parse_nodes(content)
    links = _parse_links(content)
    demands = _parse_demands(content)

    logger.info(f"Parsed {len(nodes)} nodes, {len(links)} links, {len(demands)} demands")

    return {
        'nodes': nodes,
        'links': links,
        'demands': demands
    }


def _parse_nodes(content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the NODES section from SNDlib content.

    Format: N1 ( 167.00 208.00 )

    Args:
        content: Full file content

    Returns:
        Dictionary mapping node_id to {'x': float, 'y': float}
    """
    nodes = {}

    # Find NODES section - match between "NODES (" and the next section or end
    nodes_match = re.search(r'NODES\s*\(([^)]+(?:\([^)]+\)[^)]*)*)\)(?=\s*(?:LINKS|DEMANDS|$))', content, re.DOTALL)
    if not nodes_match:
        # Try simpler pattern - find content until next section
        nodes_match = re.search(r'NODES\s*\((.*?)^\)', content, re.DOTALL | re.MULTILINE)

    if not nodes_match:
        logger.warning("No NODES section found")
        return nodes

    nodes_content = nodes_match.group(1)

    # Parse each node line: N1 ( 167.00 208.00 ) or ATLAM5 ( -84.3833 33.75 )
    # Note: Coordinates can be negative
    pattern = r'(\w+)\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)'
    for match in re.finditer(pattern, nodes_content):
        node_id = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        nodes[node_id] = {'x': x, 'y': y}

    return nodes


def _parse_links(content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the LINKS section from SNDlib content.

    Format: L1_N20_N3 ( N20 N3 ) 0.00 0.00 0.00 0.00 ( 504000.00 162160.02 ... )

    Args:
        content: Full file content

    Returns:
        Dictionary mapping link_id to link attributes
    """
    links = {}

    # Find LINKS section - find content until DEMANDS section
    links_match = re.search(r'LINKS\s*\((.*?)(?=\n\s*DEMANDS|\n\s*ADMISSIBLE_PATHS|\Z)', content, re.DOTALL)
    if not links_match:
        logger.warning("No LINKS section found")
        return links

    links_content = links_match.group(1)

    # Parse each link line
    # Pattern: link_id ( source target ) pre_cap pre_cost routing_cost setup_cost ( modules... )
    pattern = r'(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\(([\d.\s]+)\)'

    for match in re.finditer(pattern, links_content):
        link_id = match.group(1)
        source = match.group(2)
        target = match.group(3)
        pre_installed_cap = float(match.group(4))
        pre_installed_cost = float(match.group(5))
        routing_cost = float(match.group(6))
        setup_cost = float(match.group(7))
        modules_str = match.group(8)

        # Parse capacity modules (pairs of capacity and cost)
        modules = [float(x) for x in modules_str.split()]
        capacity_modules = []
        for i in range(0, len(modules), 2):
            if i + 1 < len(modules):
                capacity_modules.append({
                    'capacity': modules[i],
                    'cost': modules[i + 1]
                })

        # Determine capacity and unit cost for MCNF
        # Priority 1: Use module capacity (typically larger and more realistic)
        # Priority 2: Use pre-installed capacity
        # Priority 3: Uncapacitated (infinite capacity)
        if capacity_modules:
            # Use the largest capacity module
            max_module = max(capacity_modules, key=lambda m: m['capacity'])
            capacity = max_module['capacity']
            # Use amortized module cost as unit cost (installation cost per unit capacity)
            unit_cost = max_module['cost'] / max_module['capacity']
        elif pre_installed_cap > 0:
            # Fall back to pre-installed capacity
            capacity = pre_installed_cap
            # Use routing cost, or 0.0 if unspecified (free routing)
            unit_cost = routing_cost if routing_cost > 0 else 1.0
        else:
            # Uncapacitated network
            capacity = float('inf')
            unit_cost = routing_cost if routing_cost > 0 else 1.0

        links[link_id] = {
            'source': source,
            'target': target,
            'capacity': capacity,
            'unit_cost': unit_cost,
            'routing_cost': routing_cost,
            'capacity_modules': capacity_modules
        }

    return links


def _parse_demands(content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the DEMANDS section from SNDlib content.

    Format: D1_N2_N15 ( N2 N15 ) 1 819678.00 20
    or:     IPLSng_STTLng ( IPLSng STTLng ) 1 3580.00 UNLIMITED

    Args:
        content: Full file content

    Returns:
        Dictionary mapping demand_id to demand attributes
    """
    demands = {}

    # Find DEMANDS section - find content until ADMISSIBLE_PATHS or end
    demands_match = re.search(r'DEMANDS\s*\((.*?)(?=\n\s*ADMISSIBLE_PATHS|\Z)', content, re.DOTALL)
    if not demands_match:
        logger.warning("No DEMANDS section found")
        return demands

    demands_content = demands_match.group(1)

    # Parse each demand line - max_path_length can be a number or "UNLIMITED"
    pattern = r'(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)\s+(\d+)\s+([\d.]+)\s+(\w+)'

    for match in re.finditer(pattern, demands_content):
        demand_id = match.group(1)
        source = match.group(2)
        target = match.group(3)
        routing_unit = int(match.group(4))
        demand_value = float(match.group(5))
        max_path_str = match.group(6)

        # Convert UNLIMITED to a large number
        if max_path_str.upper() == 'UNLIMITED':
            max_path_length = 9999
        else:
            max_path_length = int(max_path_str)

        demands[demand_id] = {
            'source': source,
            'target': target,
            'routing_unit': routing_unit,
            'demand_value': demand_value,
            'max_path_length': max_path_length
        }

    return demands
