"""
Optimization Package

This package provides tools for building and solving linear programming problems.

Modules:
    modeling: Problem modeling interface (PuLP-like API)
    solvers: Mathematical solver implementations (interior-point, etc.)

Main Exports:
    From modeling:
        - LpProblem: Main class for LP problem construction
        - LpVariable_dicts: Helper for creating indexed variables
        - LPStandardForm: Standard form representation
        
    From solvers:
        - solve: Main solver entry point
        - SolverError: Solver exception class
"""

from .modeling import (
    LpProblem,
    LpVariable_dicts,
    LPStandardForm,
)

from .solvers import (
    solve,
    SolverError,
)

__all__ = [
    # Modeling
    'LpProblem',
    'LpVariable_dicts',
    'LPStandardForm',
    # Solvers
    'solve',
    'SolverError',
]