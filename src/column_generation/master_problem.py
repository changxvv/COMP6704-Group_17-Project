"""
Restricted Master Problem (RMP) for Column Generation.

This module implements the RMP using Gurobi linear programming library.
"""

from typing import Dict, List, Any
import logging
import gurobipy as gp
from gurobipy import GRB

from .data_models import Network, Edge

logger = logging.getLogger(__name__)


class MasterProblem:
    """
    Restricted Master Problem solver using Gurobi with Big-M method.

    The RMP is formulated as:
    min Σ(cost_k * λ_k) + M * Σ(artificial[e])
    s.t. Σ(flow_k[e] * λ_k) + slack[e] - artificial[e] = capacity[e]  ∀e ∈ bounded_edges
         Σ(λ_k) = 1
         λ_k, slack[e], artificial[e] ≥ 0  ∀k, e

    The artificial variables ensure the RMP is always feasible (Big-M method).
    """

    def __init__(self, network: Network, bounded_edges: Dict[Any, Edge], M: float = 1e6):
        """
        Initialize the Master Problem with Gurobi model.

        Args:
            network: The network
            bounded_edges: Dictionary of edges with capacity constraints
            M: Big-M penalty parameter for artificial variables (default: 1e6)
        """
        self.network = network
        self.bounded_edges = bounded_edges
        self.M = M

        # Create Gurobi model
        self.model = gp.Model("Restricted_Master_Problem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('LogFile', '')

        # Store variables and constraints
        self.lambda_vars = []
        self.slack_vars = {}
        self.artificial_vars = {}
        self.capacity_constrs = {}
        self.convexity_constr = None

        # Create slack and artificial variables for each bounded edge
        for edge_id in self.bounded_edges:
            edge = self.bounded_edges[edge_id]
            self.slack_vars[edge_id] = self.model.addVar(
                lb=0.0,
                obj=0.0,
                name=f"slack_{edge_id}"
            )
            self.artificial_vars[edge_id] = self.model.addVar(
                lb=0.0,
                obj=self.M,
                name=f"artificial_{edge_id}"
            )

        # Create capacity constraints (initially without lambda terms)
        # Σ(flow_k[e] * λ_k) + slack[e] - artificial[e] = capacity[e]
        for edge_id in self.bounded_edges:
            edge = self.bounded_edges[edge_id]
            constr = self.model.addConstr(
                self.slack_vars[edge_id] - self.artificial_vars[edge_id] == edge.capacity,
                name=f"capacity_{edge_id}"
            )
            self.capacity_constrs[edge_id] = constr

        # Convexity constraint will be created when first column is added
        self.convexity_constr = None

        # Update model to integrate new variables and constraints
        self.model.update()

        logger.debug(f"MasterProblem initialized with {len(self.bounded_edges)} bounded edges")

    def add_column(self, solution: Dict[str, Any]) -> None:
        """
        Add a new column (lambda variable) to the RMP dynamically.

        Args:
            solution: Solution dictionary containing 'cost', 'flow', and 'routes'
        """
        lambda_idx = len(self.lambda_vars)

        # For first column, create convexity constraint directly
        if lambda_idx == 0:
            # Add lambda variable without column
            lambda_var = self.model.addVar(
                lb=0.0,
                ub=1.0,
                obj=solution['cost'],
                name=f"lambda_{lambda_idx}"
            )
            self.lambda_vars.append(lambda_var)
            self.model.update()

            # Create convexity constraint: Σ(λ_k) = 1
            self.convexity_constr = self.model.addConstr(
                lambda_var == 1,
                name="convexity"
            )

            # Add to capacity constraints manually
            for edge_id in self.bounded_edges:
                flow_on_edge = solution['flow'].get(edge_id, 0.0)
                constr = self.capacity_constrs[edge_id]
                self.model.chgCoeff(constr, lambda_var, flow_on_edge)

            self.model.update()
            logger.debug(f"Added first column with cost {solution['cost']:.2f}")
        else:
            # For subsequent columns, use column-wise addition
            # Build column coefficients for all constraints
            col_coeffs = []
            col_constrs = []

            # Coefficients for capacity constraints
            for edge_id in self.bounded_edges:
                flow_on_edge = solution['flow'].get(edge_id, 0.0)
                col_coeffs.append(flow_on_edge)
                col_constrs.append(self.capacity_constrs[edge_id])

            # Coefficient for convexity constraint (always 1)
            col_coeffs.append(1.0)
            col_constrs.append(self.convexity_constr)

            # Create column
            column = gp.Column(col_coeffs, col_constrs)

            # Add lambda variable with column
            lambda_var = self.model.addVar(
                lb=0.0,
                ub=1.0,
                obj=solution['cost'],
                name=f"lambda_{lambda_idx}",
                column=column
            )

            self.lambda_vars.append(lambda_var)

            # Update model to integrate new variable
            self.model.update()

            logger.debug(f"Added column {lambda_idx} with cost {solution['cost']:.2f}")

    def solve(self) -> Dict[str, Any]:
        """
        Solve the RMP and return results.

        Returns:
            Dictionary with objective value, solution status, and feasibility info
        """
        logger.debug(f"Solving RMP with {len(self.lambda_vars)} columns")

        # Solve the model
        self.model.optimize()

        # Check solution status
        if self.model.status != GRB.OPTIMAL:
            logger.warning(f"RMP solution status: {self.model.status}")
            return {
                'status': 'Non-optimal',
                'objective': float('inf'),
                'total_artificial': float('inf'),
                'is_feasible': False
            }

        # Get objective value
        obj_value = self.model.ObjVal

        # Calculate total artificial variable value to check true feasibility
        total_artificial = sum(
            var.X for var in self.artificial_vars.values()
        )

        result = {
            'status': 'Optimal',
            'objective': obj_value,
            'total_artificial': total_artificial,
            'is_feasible': total_artificial < 1e-6
        }

        logger.debug(f"RMP objective: {result['objective']:.2f}, "
                    f"artificial: {total_artificial:.6f}, "
                    f"feasible: {result['is_feasible']}")
        return result

    def get_dual_variables(self) -> List[float]:
        """
        Extract dual variables from the solved RMP.

        Returns:
            List of dual variables [π_1, ..., π_m, μ]
            where π_i are duals for capacity constraints
            and μ is dual for convexity constraint
        """
        if self.model.status != GRB.OPTIMAL:
            logger.error("Model not optimally solved yet")
            return []

        dual_vars = []

        # Get duals from capacity constraints
        for edge_id in self.bounded_edges:
            constr = self.capacity_constrs[edge_id]
            dual_value = constr.Pi
            dual_vars.append(dual_value)

        # Get dual from convexity constraint
        dual_value = self.convexity_constr.Pi
        dual_vars.append(dual_value)

        return dual_vars

    def get_lambda_values(self) -> Dict[int, float]:
        """
        Get non-zero lambda variable values from the solution.

        Returns:
            Dictionary mapping column index to lambda value
        """
        if self.model.status != GRB.OPTIMAL:
            logger.error("Model not optimally solved yet")
            return {}

        lambda_values = {}
        for idx, var in enumerate(self.lambda_vars):
            value = var.X
            if value > 1e-9:
                lambda_values[idx] = value

        return lambda_values

    def get_objective_value(self) -> float:
        """
        Get the objective value of the solved problem.

        Returns:
            Objective value
        """
        if self.model.status != GRB.OPTIMAL:
            return 0.0
        return self.model.ObjVal
