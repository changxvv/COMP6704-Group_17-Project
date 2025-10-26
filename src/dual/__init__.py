"""
Dual Decomp Method package for Multi-Commodity Network Flow.

This package provides a clean, modern implementation of the Dual Decomp Method
for solving MCNF problems using edge-based LP formulation.
"""

from .data_models import Node, Edge, Demand, Network
from .dual_decomp import DualDecomposition
from .parser import parse_sndlib_file
from .lmcf_parser import parse_lmcf_files

__all__ = [
    'Node',
    'Edge',
    'Demand',
    'Network',
    'DualDecomposition',
    'parse_sndlib_file',
    'parse_lmcf_files'
]