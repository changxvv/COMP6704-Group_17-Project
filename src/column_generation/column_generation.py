"""
Column Generation Algorithm Controller.

This module implements the Dantzig-Wolfe decomposition algorithm
for Multi-Commodity Network Flow problems.
"""

import time
from pathlib import Path
from typing import Dict, List, Any
import logging

from .data_models import Network
from .master_problem import MasterProblem
from .pricing_problem import generate_initial_solution, solve_pricing_problem
from .utils import get_bounded_edges, reset_edge_capacity, reset_edge_weight

logger = logging.getLogger(__name__)


class ColumnGeneration:
    """
    Column Generation algorithm controller using Dantzig-Wolfe decomposition.

    Solves Multi-Commodity Network Flow problems by iteratively solving:
    1. Restricted Master Problem (RMP) to get dual variables
    2. Pricing subproblem to generate new columns with negative reduced cost
    """

    def __init__(
        self,
        network: Network,
        epsilon: float = 1e-6,
        max_iterations: int = 2000,
        output_folder: str = "./output/",
        M: float = 1e6
    ):
        """
        Initialize Column Generation algorithm.

        Args:
            network: The network to optimize
            epsilon: Convergence tolerance for reduced cost
            max_iterations: Maximum number of iterations
            output_folder: Folder for output files
            M: Big-M penalty parameter for artificial variables (default: 1e6)
        """
        self.network = network
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.M = M

        self.solutions: List[Dict[str, Any]] = []
        self.bounded_edges: Dict = {}
        self.master_problem: MasterProblem = None
        self.iteration_history: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """
        Run the Column Generation algorithm with Big-M method.

        Returns:
            Dictionary with final results including objective value and solve time
        """
        logger.info("="*60)
        logger.info("Starting Column Generation Algorithm")
        logger.info("="*60)

        start_time = time.time()

        # Find edges with bounded capacity
        self.bounded_edges = get_bounded_edges(self.network)

        # Generate initial solution (extreme point)
        logger.info("Generating initial solution...")
        initial_solution = generate_initial_solution(self.network)
        self.solutions.append(initial_solution)

        # Create master problem with Big-M
        self.master_problem = MasterProblem(self.network, self.bounded_edges, M=self.M)

        # Add initial column to the model
        self.master_problem.add_column(initial_solution)

        # Main column generation loop
        iteration = 0
        is_feasible = False
        reduced_cost = float('inf')

        while iteration < self.max_iterations:
            # Solve RMP
            result = self.master_problem.solve()
            objective = result['objective']
            is_feasible = result['is_feasible']
            total_artificial = result['total_artificial']

            # Get dual variables
            dual_vars = self.master_problem.get_dual_variables()

            # Solve pricing problem
            new_solution, reduced_cost = solve_pricing_problem(
                self.network, dual_vars, self.bounded_edges
            )

            # Record iteration
            self.iteration_history.append({
                'iteration': iteration,
                'objective': objective,
                'reduced_cost': reduced_cost,
                'num_columns': len(self.solutions),
                'total_artificial': total_artificial,
                'is_feasible': is_feasible
            })

            # Log progress
            if iteration % 10 == 0 or is_feasible:
                logger.info(f"Iter {iteration}: obj={objective:.2f}, "
                          f"reduced_cost={reduced_cost:.6f}, "
                          f"artificial={total_artificial:.6f}, "
                          f"feasible={is_feasible}")

            # FIXED CONVERGENCE LOGIC:
            # Stop only when BOTH conditions met:
            # 1. Solution is truly feasible (artificial = 0)
            # 2. No improving columns (reduced_cost <= epsilon)
            if is_feasible and reduced_cost <= self.epsilon:
                logger.info("Converged: feasible solution with no improving columns")
                break

            # Add column if:
            # - It improves (reduced_cost > epsilon), OR
            # - Still in Phase I (not feasible yet)
            if reduced_cost > self.epsilon or not is_feasible:
                self.solutions.append(new_solution)
                self.master_problem.add_column(new_solution)
            else:
                # Feasible and no improving column
                break

            iteration += 1

        end_time = time.time()
        solve_time = end_time - start_time

        # Check if we hit max iterations without converging
        if iteration >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            if not is_feasible:
                logger.error("Problem may be infeasible - artificial variables still positive")

        # Final results
        final_objective = self.master_problem.get_objective_value()

        logger.info("="*60)
        logger.info("Column Generation Completed")
        logger.info(f"Iterations: {iteration}")
        logger.info(f"Final objective: {final_objective:.2f}")
        logger.info(f"Total columns generated: {len(self.solutions)}")
        logger.info(f"Solve time: {solve_time:.2f} seconds")
        logger.info("="*60)

        return {
            'objective': final_objective,
            'iterations': iteration,
            'num_columns': len(self.solutions),
            'solve_time': solve_time,
            'converged': is_feasible and reduced_cost <= self.epsilon,
            'is_feasible': is_feasible
        }

    def retrieve_solution(self) -> Dict[str, Any]:
        """
        Extract final routes and flows from lambda values.

        Returns:
            Dictionary with routes and flows for each demand
        """
        logger.info("Retrieving final solution...")

        # Reset network state
        reset_edge_capacity(self.network)
        reset_edge_weight(self.network)

        # Get lambda values
        lambda_values = self.master_problem.get_lambda_values()

        # For each non-zero lambda, update demand routes and edge flows
        for sol_idx, ratio in lambda_values.items():
            solution = self.solutions[sol_idx]
            routes_dict = solution['routes']

            for demand_id, route in routes_dict.items():
                demand = self.network.demands[demand_id]
                demand.update_route(route, ratio)

                # Update edge flows
                for i in range(len(route) - 1):
                    u_id = route[i].id
                    v_id = route[i + 1].id
                    edge_id = self.network.edge_dict[(u_id, v_id)]
                    edge = self.network.edges[edge_id]
                    edge.use_capacity(demand.quantity * ratio)

        logger.info("Solution retrieval complete")

        return {
            'lambda_values': lambda_values,
            'num_routes': sum(len(d.routes) for d in self.network.demands.values())
        }

    def save_results(self) -> None:
        """
        Save results to output files.

        Creates:
        - iterations.txt: iteration history
        - variables.txt: lambda values
        - routes.txt: routes for each demand
        - flow.txt: flow on each edge
        """
        logger.info(f"Saving results to {self.output_folder}")

        # Save iteration history
        with open(self.output_folder / "iterations.txt", 'w') as f:
            f.write("iter_num\tobj\treduced_cost\tnum_columns\tartificial\tfeasible\n")
            for record in self.iteration_history:
                f.write(f"{record['iteration']}\t{record['objective']:.2f}\t"
                       f"{record['reduced_cost']:.6f}\t{record['num_columns']}\t"
                       f"{record['total_artificial']:.6f}\t{record['is_feasible']}\n")

        # Save lambda values
        lambda_values = self.master_problem.get_lambda_values()
        with open(self.output_folder / "variables.txt", 'w') as f:
            f.write("var_name\tvalue\n")
            for k, value in lambda_values.items():
                f.write(f"lambda_{k}\t{value:.6f}\n")

        # Save routes
        with open(self.output_folder / "routes.txt", 'w') as f:
            f.write("demand_id\tratio\troute\n")
            for demand in self.network.demands.values():
                for route_hash, ratio in demand.routes.items():
                    f.write(f"{demand.id}\t{ratio:.6f}\t{route_hash}\n")

        # Save edge flows
        with open(self.output_folder / "flow.txt", 'w') as f:
            f.write("edge_id\tsource\ttarget\tflow\tcapacity\n")
            for edge in self.network.edges.values():
                f.write(f"{edge.id}\t{edge.u.id}\t{edge.v.id}\t"
                       f"{edge.capacity_used:.2f}\t{edge.capacity:.2f}\n")

        logger.info("Results saved successfully")
