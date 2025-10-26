"""
Entry point for Column Generation algorithm.

This script provides a command-line interface for solving
Multi-Commodity Network Flow problems using Column Generation.
"""

import argparse
import logging
import sys
from pathlib import Path

from .data_models import Network
from .column_generation import ColumnGeneration


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Solve Multi-Commodity Network Flow using Column Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input file: SNDlib format (e.g., tests/ta1.txt) or LMCF network file (e.g., tests/LMCF/GridDemands/Cgd1.txt)'
    )

    parser.add_argument(
        '--lmcf',
        action='store_true',
        help='Use LMCF format (requires --demand-file). Input file should be network file (C*.txt).'
    )

    parser.add_argument(
        '--demand-file',
        type=str,
        default=None,
        help='Demand file for LMCF format (D*.txt). Required when --lmcf is set.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output/',
        help='Output folder for results'
    )

    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=1e-6,
        help='Convergence tolerance for reduced cost'
    )

    parser.add_argument(
        '-m', '--max-iter',
        type=int,
        default=2000,
        help='Maximum number of iterations'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (debug) logging'
    )

    parser.add_argument(
        '--directed',
        action='store_true',
        help='Treat links as directed (do not create reverse edges). '
             'Use this if the input file already has bidirectional links.'
    )

    parser.add_argument(
        '-M', '--big-m',
        type=float,
        default=1e9,
        help='Big-M penalty parameter for artificial variables (default: 1e6)'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the Column Generation algorithm.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        # Validate LMCF mode requirements
        if args.lmcf:
            if not args.demand_file:
                logger.error("--demand-file is required when using --lmcf mode")
                return 1
            demand_path = Path(args.demand_file)
            if not demand_path.exists():
                logger.error(f"Demand file not found: {demand_path}")
                return 1

        # Load network
        network = Network()
        if args.lmcf:
            logger.info(f"Loading LMCF network from: {input_path.name} and {demand_path.name}")
            network.load_from_lmcf(input_path, demand_path, undirected=not args.directed)
        else:
            logger.info(f"Loading SNDlib network from: {input_path}")
            network.load_from_sndlib(input_path, undirected=not args.directed)

        # Create column generation solver
        cg = ColumnGeneration(
            network=network,
            epsilon=args.epsilon,
            max_iterations=args.max_iter,
            output_folder=args.output,
            M=args.big_m
        )

        # Run algorithm
        results = cg.run()

        # Retrieve solution
        cg.retrieve_solution()

        # Save results
        cg.save_results()

        # Print summary
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        print(f"Objective value:     {results['objective']:.2f}")
        print(f"Iterations:          {results['iterations']}")
        print(f"Columns generated:   {results['num_columns']}")
        print(f"Solve time:          {results['solve_time']:.2f} seconds")
        print(f"Converged:           {'Yes' if results['converged'] else 'No'}")
        print(f"Output folder:       {args.output}")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
