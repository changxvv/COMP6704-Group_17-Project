"""
Entry point for Column Generation algorithm (Dual Decomposition backend).
"""

import argparse
import logging
import sys
from pathlib import Path

from .data_models import Network
from .dual_decomp import DualDecomposition


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve Multi-Commodity Network Flow using Column Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file (SNDlib) or LMCF network file')
    parser.add_argument('--lmcf', action='store_true', help='Use LMCF format')
    parser.add_argument('--demand-file', type=str, default=None,
                        help='Demand file for LMCF format (D*.txt)')
    parser.add_argument('-o', '--output', type=str, default='./output/', help='Output folder')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('-m', '--max-iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--directed', action='store_true',
                        help='Treat links as directed (do not create reverse edges)')
    parser.add_argument('-M', '--big-m', type=float, default=1e9, help='Big-M penalty')
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        if args.lmcf:
            if not args.demand_file:
                logger.error("--demand-file is required when using --lmcf mode")
                return 1
            demand_path = Path(args.demand_file)
            if not demand_path.exists():
                logger.error(f"Demand file not found: {demand_path}")
                return 1

        network = Network()
        if args.lmcf:
            logger.info(f"Loading LMCF network from: {input_path.name} and {demand_path.name}")
            network.load_from_lmcf(input_path, demand_path, undirected=not args.directed)
        else:
            logger.info(f"Loading SNDlib network from: {input_path}")
            network.load_from_sndlib(input_path, undirected=not args.directed)

        dual = DualDecomposition(
            network=network,
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            output_folder=args.output,
            M=args.big_m
        )

        results = dual.run()
        dual.save_results()  # 再次调用安全（主要确保文件都就位）

        # ---- 打印更详细的“解状态 + 质量” ----
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        # 兼容字段名
        time_sec = results.get('solve_time', 0.0)
        objective = results.get('objective', float('nan'))
        status = results.get('status', 'unknown')
        iterations = results.get('iterations', 0)
        primal_residual = results.get('primal_residual', None)
        dual_residual = results.get('dual_residual', None)
        complementarity = results.get('complementarity', None)

        print(f"  求解时间: {time_sec:.4f}秒")
        print(f"  目标值: {objective:.6f}")
        print(f"  终止原因: {status}")
        print(f"  收敛状态: {'成功' if status in ('optimal', 'optimal_stagnated') else '未成功'}")
        print(f"  迭代次数: {iterations}")
        print("  解的质量:")
        if primal_residual is not None:
            print(f"    原始可行性残差: {primal_residual:.2e} {'[WARNING]' if primal_residual > 1e-6 else '[OK]'}")
        if dual_residual is not None:
            print(f"    对偶可行性残差: {dual_residual:.2e} {'[OK]' if dual_residual < 1e-6 else '[WARNING]'}")
        if complementarity is not None:
            print(f"    互补性: {complementarity:.2e} {'[OK]' if complementarity < 1e-6 else '[WARNING]'}")

        print("-"*60)
        print(f"  输出目录: {args.output}")
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
