"""
Benchmark runner for SNDlib and LMCF network test suites.

This script runs the Column Generation algorithm on network datasets
and collects performance metrics for analysis.
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fnmatch


# Networks that should be treated as directed
DIRECTED_NETWORKS = {
    'giul39',
    'janos-us',
    'janos-us-ca',
    'sun'
}


def discover_lmcf_pairs(base_dir: Path) -> List[Dict[str, any]]:
    """
    Discover and pair LMCF network and demand files.

    Args:
        base_dir: Base directory containing LMCF files (e.g., tests/LMCF)

    Returns:
        List of dictionaries with network_file, demand_file, and metadata
    """
    pairs = []

    # Recursively find all C*.txt files
    network_files = sorted(base_dir.rglob('C*.txt'))

    for network_file in network_files:
        # Extract the suffix (everything after 'C')
        filename = network_file.stem  # e.g., 'Cgd1'
        if not filename.startswith('C'):
            continue

        suffix = filename[1:]  # e.g., 'gd1'

        # Look for matching D*.txt in the same directory
        demand_file = network_file.parent / f"D{suffix}.txt"

        if demand_file.exists():
            # Determine category from directory
            category = network_file.parent.name
            if category == base_dir.name:
                category = "Other"

            pairs.append({
                'network_file': network_file,
                'demand_file': demand_file,
                'problem_name': suffix,
                'category': category
            })
        else:
            print(f"Warning: No matching demand file for {network_file.name}")

    return pairs


def parse_network_stats(filepath: Path) -> Tuple[int, int, int]:
    """
    Parse SNDlib file to extract network statistics.

    Args:
        filepath: Path to SNDlib network file

    Returns:
        Tuple of (num_nodes, num_links, num_demands)
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Count nodes in NODES section
        # Match from NODES ( to ) at line start (section closing paren)
        nodes_section = re.search(r'NODES\s*\((.*?)^\s*\)', content, re.DOTALL | re.MULTILINE)
        num_nodes = 0
        if nodes_section:
            # Count lines that start with an identifier
            node_pattern = re.compile(r'^\s*([a-zA-Z0-9_-]+)\s+\(', re.MULTILINE)
            num_nodes = len(node_pattern.findall(nodes_section.group(1)))

        # Count links in LINKS section
        links_section = re.search(r'LINKS\s*\((.*?)^\s*\)', content, re.DOTALL | re.MULTILINE)
        num_links = 0
        if links_section:
            # Count lines that start with a link ID
            link_pattern = re.compile(r'^\s*([a-zA-Z0-9_-]+)\s+\(', re.MULTILINE)
            num_links = len(link_pattern.findall(links_section.group(1)))

        # Count demands in DEMANDS section
        demands_section = re.search(r'DEMANDS\s*\((.*?)^\s*\)', content, re.DOTALL | re.MULTILINE)
        num_demands = 0
        if demands_section:
            # Count lines that start with a demand ID
            demand_pattern = re.compile(r'^\s*([a-zA-Z0-9_-]+)\s+\(', re.MULTILINE)
            num_demands = len(demand_pattern.findall(demands_section.group(1)))

        return num_nodes, num_links, num_demands

    except Exception as e:
        print(f"Warning: Could not parse network stats from {filepath.name}: {e}")
        return 0, 0, 0


def parse_output(stdout: str) -> Dict[str, any]:
    """
    Parse the output from main.py to extract metrics.

    Args:
        stdout: Standard output from the solver

    Returns:
        Dictionary with parsed metrics
    """
    result = {
        'objective': None,
        'iterations': None,
        'columns': None,
        'time_seconds': None,
        'converged': None
    }

    try:
        # Extract objective value
        obj_match = re.search(r'Objective value:\s+([\d.]+)', stdout)
        if obj_match:
            result['objective'] = float(obj_match.group(1))

        # Extract iterations
        iter_match = re.search(r'Iterations:\s+(\d+)', stdout)
        if iter_match:
            result['iterations'] = int(iter_match.group(1))

        # Extract columns
        col_match = re.search(r'Columns generated:\s+(\d+)', stdout)
        if col_match:
            result['columns'] = int(col_match.group(1))

        # Extract solve time
        time_match = re.search(r'Solve time:\s+([\d.]+)\s+seconds', stdout)
        if time_match:
            result['time_seconds'] = float(time_match.group(1))

        # Extract convergence
        conv_match = re.search(r'Converged:\s+(Yes|No)', stdout)
        if conv_match:
            result['converged'] = conv_match.group(1) == 'Yes'

    except Exception as e:
        print(f"Warning: Error parsing output: {e}")

    return result


def run_single_test(network_path: Path, is_directed: bool, timeout: int,
                   verbose: bool = False, demand_file: Path = None, is_lmcf: bool = False) -> Dict[str, any]:
    """
    Run the Column Generation algorithm on a single network.

    Args:
        network_path: Path to network file
        is_directed: Whether network is directed
        timeout: Timeout in seconds
        verbose: Enable verbose output
        demand_file: Path to demand file (for LMCF format)
        is_lmcf: Whether this is LMCF format

    Returns:
        Dictionary with test results
    """
    network_name = network_path.stem

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build command
        cmd = [
            sys.executable, '-m', 'src.column_generation.main',
            '-i', str(network_path),
            '-o', temp_dir,
        ]

        if is_lmcf:
            cmd.append('--lmcf')
            if demand_file:
                cmd.extend(['--demand-file', str(demand_file)])

        if is_directed:
            cmd.append('--directed')

        # Run the test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent  # Project root
            )
            elapsed_time = time.time() - start_time

            if verbose:
                print(f"\n--- Output for {network_name} ---")
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                print("---")

            # Check return code
            if result.returncode != 0:
                return {
                    'network_name': network_name,
                    'status': 'error',
                    'error_message': f"Exit code {result.returncode}: {result.stderr[:200]}",
                    'time_seconds': elapsed_time
                }

            # Parse output
            metrics = parse_output(result.stdout)

            return {
                'network_name': network_name,
                'status': 'success',
                'objective': metrics['objective'],
                'iterations': metrics['iterations'],
                'columns': metrics['columns'],
                'time_seconds': metrics['time_seconds'] or elapsed_time,
                'converged': metrics['converged'],
                'error_message': None
            }

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            return {
                'network_name': network_name,
                'status': 'timeout',
                'error_message': f'Timeout after {timeout} seconds',
                'time_seconds': elapsed_time
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                'network_name': network_name,
                'status': 'error',
                'error_message': str(e)[:200],
                'time_seconds': elapsed_time
            }


def run_all_tests(networks_dir: Path, output_file: Path, timeout: int,
                 pattern: Optional[str] = None, verbose: bool = False) -> List[Dict]:
    """
    Run tests on all networks in the directory.

    Args:
        networks_dir: Directory containing network files
        output_file: Path to output CSV file
        timeout: Timeout per test in seconds
        pattern: Optional pattern to filter networks (e.g., "nobel-*")
        verbose: Enable verbose output

    Returns:
        List of result dictionaries
    """
    # Find all .txt files in the directory
    all_networks = sorted(networks_dir.glob('*.txt'))

    # Filter by pattern if provided
    if pattern:
        all_networks = [n for n in all_networks if fnmatch.fnmatch(n.stem, pattern)]

    if not all_networks:
        print(f"No networks found in {networks_dir}")
        if pattern:
            print(f"Pattern: {pattern}")
        return []

    print(f"\nFound {len(all_networks)} networks to test")
    print("="*60)

    results = []

    for i, network_path in enumerate(all_networks, 1):
        network_name = network_path.stem

        # Determine if directed
        is_directed = network_name.lower() in DIRECTED_NETWORKS
        direction_str = "directed" if is_directed else "undirected"

        # Parse network stats
        num_nodes, num_links, num_demands = parse_network_stats(network_path)

        print(f"\n[{i}/{len(all_networks)}] Testing: {network_name} ({direction_str})")
        print(f"    Stats: {num_nodes} nodes, {num_links} links, {num_demands} demands")
        print(f"    Timeout: {timeout}s")

        # Run the test
        test_result = run_single_test(network_path, is_directed, timeout, verbose)

        # Add network stats to result
        test_result['is_directed'] = is_directed
        test_result['nodes'] = num_nodes
        test_result['links'] = num_links
        test_result['demands'] = num_demands

        # Print result
        if test_result['status'] == 'success':
            print(f"    ✓ Success: obj={test_result['objective']:.2f}, "
                  f"iters={test_result['iterations']}, "
                  f"time={test_result['time_seconds']:.2f}s")
        elif test_result['status'] == 'timeout':
            print(f"    ✗ Timeout after {timeout}s")
        else:
            print(f"    ✗ Error: {test_result['error_message'][:80]}")

        results.append(test_result)

    return results


def run_lmcf_tests(lmcf_dir: Path, output_file: Path, timeout: int,
                   pattern: Optional[str] = None, verbose: bool = False) -> List[Dict]:
    """
    Run tests on all LMCF network pairs in the directory.

    Args:
        lmcf_dir: Directory containing LMCF files
        output_file: Path to output CSV file
        timeout: Timeout per test in seconds
        pattern: Optional pattern to filter networks (e.g., "gd*" for Grid problems)
        verbose: Enable verbose output

    Returns:
        List of result dictionaries
    """
    # Discover all LMCF pairs
    pairs = discover_lmcf_pairs(lmcf_dir)

    # Filter by pattern if provided
    if pattern:
        pairs = [p for p in pairs if fnmatch.fnmatch(p['problem_name'], pattern)]

    if not pairs:
        print(f"No LMCF pairs found in {lmcf_dir}")
        if pattern:
            print(f"Pattern: {pattern}")
        return []

    print(f"\nFound {len(pairs)} LMCF network pairs to test")
    print("="*60)

    results = []

    for i, pair in enumerate(pairs, 1):
        problem_name = pair['problem_name']
        network_file = pair['network_file']
        demand_file = pair['demand_file']
        category = pair['category']

        # Parse network stats (count lines and unique nodes in files)
        try:
            # Count unique nodes from network file (nodes appear in columns 0 and 1)
            nodes_set = set()
            num_links = 0
            with open(network_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            nodes_set.add(parts[0])  # source node
                            nodes_set.add(parts[1])  # target node
                            num_links += 1

            num_nodes = len(nodes_set)

            # Count demands
            with open(demand_file) as f:
                num_demands = sum(1 for line in f if line.strip() and not line.startswith('#'))
        except:
            num_nodes, num_links, num_demands = 0, 0, 0

        print(f"\n[{i}/{len(pairs)}] Testing: {problem_name} ({category})")
        print(f"    Stats: {num_nodes} nodes, {num_links} links, {num_demands} demands")
        print(f"    Files: {network_file.name}, {demand_file.name}")
        print(f"    Timeout: {timeout}s")

        # Run the test (all LMCF networks are undirected)
        test_result = run_single_test(
            network_file,
            is_directed=False,
            timeout=timeout,
            verbose=verbose,
            demand_file=demand_file,
            is_lmcf=True
        )

        # Add metadata to result
        test_result['network_name'] = problem_name
        test_result['category'] = category
        test_result['is_directed'] = False
        test_result['nodes'] = num_nodes
        test_result['links'] = num_links
        test_result['demands'] = num_demands

        # Print result
        if test_result['status'] == 'success':
            print(f"    ✓ Success: obj={test_result['objective']:.2f}, "
                  f"iters={test_result['iterations']}, "
                  f"time={test_result['time_seconds']:.2f}s")
        elif test_result['status'] == 'timeout':
            print(f"    ✗ Timeout after {timeout}s")
        else:
            print(f"    ✗ Error: {test_result['error_message'][:80]}")

        results.append(test_result)

    return results


def save_results(results: List[Dict], output_file: Path) -> None:
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return

    # Define CSV columns
    fieldnames = [
        'network_name', 'is_directed', 'nodes', 'links', 'demands',
        'objective', 'iterations', 'columns', 'time_seconds',
        'converged', 'status', 'error_message'
    ]

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Ensure all fields are present
            row = {field: result.get(field, None) for field in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def print_summary(results: List[Dict]) -> None:
    """
    Print summary statistics.

    Args:
        results: List of result dictionaries
    """
    if not results:
        return

    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    timeouts = sum(1 for r in results if r['status'] == 'timeout')
    errors = sum(1 for r in results if r['status'] == 'error')

    total_time = sum(r.get('time_seconds', 0) or 0 for r in results)

    # Calculate averages for successful runs
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_iterations = sum(r['iterations'] for r in successful_results) / len(successful_results)
        avg_time = sum(r['time_seconds'] for r in successful_results) / len(successful_results)
        avg_columns = sum(r['columns'] for r in successful_results) / len(successful_results)
    else:
        avg_iterations = avg_time = avg_columns = 0

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total networks:      {total}")
    print(f"Successful:          {successful} ({successful/total*100:.1f}%)")
    print(f"Timeouts:            {timeouts} ({timeouts/total*100:.1f}%)")
    print(f"Errors:              {errors} ({errors/total*100:.1f}%)")
    print(f"Total time:          {total_time:.2f} seconds")

    if successful_results:
        print(f"\nAverages (successful runs):")
        print(f"  Iterations:        {avg_iterations:.1f}")
        print(f"  Columns:           {avg_columns:.1f}")
        print(f"  Time:              {avg_time:.2f} seconds")

    # Print top 5 by time
    if successful_results:
        print(f"\nTop 5 slowest networks:")
        sorted_by_time = sorted(successful_results,
                               key=lambda x: x['time_seconds'],
                               reverse=True)[:5]
        for i, r in enumerate(sorted_by_time, 1):
            print(f"  {i}. {r['network_name']}: {r['time_seconds']:.2f}s "
                  f"({r['iterations']} iters)")

    print("="*60)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run Column Generation benchmark on SNDlib or LMCF networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['sndlib', 'lmcf'],
        default=None,
        help='Dataset type (auto-detected if not specified)'
    )

    parser.add_argument(
        '--networks-dir',
        type=Path,
        default=None,
        help='Directory containing network files (default: tests/SNDlib or tests/LMCF based on dataset)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results.csv'),
        help='Output CSV file for results'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout per network in seconds'
    )

    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Pattern to filter networks (e.g., "nobel-*" or "abil*")'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (show solver output)'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for benchmark runner.

    Returns:
        Exit code (0 for success)
    """
    args = parse_arguments()

    # Auto-detect dataset type if not specified
    if args.dataset is None:
        if args.networks_dir:
            # Check directory name to guess dataset
            dir_name = args.networks_dir.name.lower()
            if 'lmcf' in dir_name:
                args.dataset = 'lmcf'
            else:
                args.dataset = 'sndlib'
        else:
            # Default to SNDlib
            args.dataset = 'sndlib'

    # Set default directory if not specified
    if args.networks_dir is None:
        if args.dataset == 'lmcf':
            args.networks_dir = Path('tests/LMCF')
        else:
            args.networks_dir = Path('tests/SNDlib')

    # Validate networks directory
    if not args.networks_dir.exists():
        print(f"Error: Networks directory not found: {args.networks_dir}")
        return 1

    print("\nColumn Generation Benchmark Runner")
    print("="*60)
    print(f"Dataset:             {args.dataset.upper()}")
    print(f"Networks directory:  {args.networks_dir}")
    print(f"Output file:         {args.output}")
    print(f"Timeout per test:    {args.timeout} seconds")
    if args.filter:
        print(f"Filter pattern:      {args.filter}")

    # Run tests based on dataset type
    start_time = time.time()
    if args.dataset == 'lmcf':
        results = run_lmcf_tests(
            args.networks_dir,
            args.output,
            args.timeout,
            args.filter,
            args.verbose
        )
    else:
        results = run_all_tests(
            args.networks_dir,
            args.output,
            args.timeout,
            args.filter,
            args.verbose
        )
    total_time = time.time() - start_time

    if not results:
        return 1

    # Save results
    save_results(results, args.output)

    # Print summary
    print_summary(results)

    print(f"\nTotal benchmark time: {total_time:.2f} seconds")

    return 0


if __name__ == '__main__':
    sys.exit(main())
