"""
Column Generation for Multi-Commodity Network Flow.

A modern Python implementation of Dantzig-Wolfe decomposition
for solving MCNF problems using Gurobi.
"""

from .data_models import Node, Edge, Demand, Network
from .column_generation import ColumnGeneration
from .parser import parse_sndlib_file

__version__ = "1.0.0"

__all__ = [
    'Node',
    'Edge',
    'Demand',
    'Network',
    'ColumnGeneration',
    'parse_sndlib_file'
]
