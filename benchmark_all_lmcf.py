#!/usr/bin/env python3
"""
Comprehensive LMCF Benchmark Script for All Implementations

This script runs all 4 implementations (Simplex, Column Generation,
Dual Decomposition, Interior Point) on all LMCF instances, executing
each instance multiple times for statistical reliability.

Author: COMP6704 Team
Date: 2025-01-21
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

LMCF_ROOT = Path(__file__).parent / "tests" / "LMCF"
PROJECT_ROOT = Path(__file__).parent

IMPLEMENTATIONS = ["simplex", "column_generation", "dual", "interior_point"]
CATEGORIES = ["GridDemands", "PlanarNetworks", "TrafficNetworks", "OtherDemands"]


# ============================================================================
# Helper Functions for Output Management
# ============================================================================

def create_run_directory(output_dir: Path, instance_name: str, implementation: str, run_number: int) -> Path:
    """Create directory for storing run artifacts."""
    safe_name = instance_name.replace('/', '_')
    run_dir = output_dir / "runs" / f"{safe_name}_{implementation}_run{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_console_output(run_dir: Path, stdout: str, stderr: str):
    """Save stdout and stderr to files."""
    (run_dir / "stdout.txt").write_text(stdout, encoding='utf-8')
    (run_dir / "stderr.txt").write_text(stderr, encoding='utf-8')


def copy_generated_files(source_dir: Path, run_dir: Path):
    """Copy all generated files from source to run directory."""
    import shutil
    for file in source_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, run_dir / file.name)


# ============================================================================
# LMCF Instance Discovery
# ============================================================================

def natural_sort_key(path: Path) -> List:
    """Natural sorting key for filenames (e.g., Cgd1, Cgd2, ..., Cgd10)"""
    name = path.name
    parts = re.split(r'(\d+)', name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_lmcf_instances(category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Auto-discover all LMCF instance pairs (C*.txt, D*.txt).

    Args:
        category_filter: Optional category name to filter instances

    Returns:
        List of instance dictionaries with metadata
    """
    instances = []
    instance_id = 1

    categories = [category_filter] if category_filter else CATEGORIES

    for category in categories:
        category_path = LMCF_ROOT / category
        if not category_path.exists():
            print(f"Warning: Category path not found: {category_path}")
            continue

        # Find all C*.txt network files
        network_files = sorted(category_path.glob("C*.txt"), key=natural_sort_key)

        for network_file in network_files:
            # Derive demand file name: C*.txt -> D*.txt
            network_name = network_file.name
            demand_name = "D" + network_name[1:]
            demand_file = category_path / demand_name

            if not demand_file.exists():
                # Special handling for Dpl2500 variants
                if network_name == "Cpl2500.txt":
                    for variant in ["Dpl2500.txt", "Dpl2500_1.txt", "Dpl2500_2.txt"]:
                        demand_file_variant = category_path / variant
                        if demand_file_variant.exists():
                            demand_name = variant
                            demand_file = demand_file_variant
                            break

                if not demand_file.exists():
                    print(f"Warning: No matching demand file for {network_file.name}")
                    continue

            # Create instance name
            base_name = network_name.replace(".txt", "").replace("C", "")
            instance_name = f"{category}/{base_name}"

            instances.append({
                "id": instance_id,
                "name": instance_name,
                "category": category,
                "network_file": str(network_file),
                "demand_file": str(demand_file),
                "network_name": network_name,
                "demand_name": demand_name,
            })
            instance_id += 1

    return instances


# ============================================================================
# Implementation Runners
# ============================================================================

def run_simplex(network_file: str, demand_file: str, time_limit: float, run_dir: Path) -> Dict[str, Any]:
    """Run Simplex method and parse output."""
    cmd = [
        sys.executable, "-m", "src.simplex.main_sparse",
        network_file,
        "--format", "lmcf",
        "--demand-file", demand_file,
        "--time-limit", str(time_limit)
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 2,  # Add buffer for initialization
            cwd=PROJECT_ROOT
        )
        elapsed = time.time() - start_time

        # Save console output
        save_console_output(run_dir, result.stdout, result.stderr)

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Exit code {result.returncode}",
                "time": elapsed,
                "run_directory": str(run_dir)
            }

        # Parse output
        parsed = parse_simplex_output(result.stdout, elapsed)
        parsed["run_directory"] = str(run_dir)
        return parsed

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Timeout after {time_limit}s",
            "time": time_limit,
            "run_directory": str(run_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
            "run_directory": str(run_dir)
        }


def parse_simplex_output(stdout: str, elapsed: float) -> Dict[str, Any]:
    """Parse Simplex console output (supports both main.py and main_sparse.py)."""
    result = {"status": "unknown", "time": elapsed}

    # Try main_sparse.py format first ([sparse] prefix)
    obj_match = re.search(r'\[sparse\] objective=([0-9.]+)', stdout)
    if obj_match:
        result["objective"] = float(obj_match.group(1))

    time_match = re.search(r'\[sparse\] Solve time:\s*([0-9.]+)s', stdout)
    if time_match:
        result["time"] = float(time_match.group(1))

    iter_match = re.search(r'\[sparse\] iterations=(\d+)', stdout)
    if iter_match:
        result["iterations"] = int(iter_match.group(1))

    # Check success from main_sparse.py
    if re.search(r'\[sparse\] success=True', stdout):
        result["status"] = "success"
        result["converged"] = True
    elif re.search(r'\[sparse\] success=False', stdout):
        result["status"] = "error"
        result["converged"] = False

    # Fallback: Try Chinese format (from main.py)
    if "objective" not in result:
        obj_match = re.search(r'目标值.*?:\s*([0-9.]+)', stdout)
        if obj_match:
            result["objective"] = float(obj_match.group(1))

    if "time" not in result or result["time"] == elapsed:
        time_match = re.search(r'求解时间.*?:\s*([0-9.]+)', stdout)
        if time_match:
            result["time"] = float(time_match.group(1))

    if "iterations" not in result:
        iter_match = re.search(r'迭代次数.*?:\s*(\d+)', stdout)
        if iter_match:
            result["iterations"] = int(iter_match.group(1))

    # Extract residuals (Chinese format only, sparse doesn't provide these)
    primal_match = re.search(r'原始残差.*?:\s*([0-9.e+-]+)', stdout)
    if primal_match:
        result["primal_residual"] = float(primal_match.group(1))

    dual_match = re.search(r'对偶残差.*?:\s*([0-9.e+-]+)', stdout)
    if dual_match:
        result["dual_residual"] = float(dual_match.group(1))

    comp_match = re.search(r'互补性.*?:\s*([0-9.e+-]+)', stdout)
    if comp_match:
        result["complementarity"] = float(comp_match.group(1))

    # Determine status from Chinese format if not already set
    if "converged" not in result:
        if "optimal" in stdout.lower():
            result["status"] = "success"
            result["converged"] = True
        elif "error" in stdout.lower():
            result["status"] = "error"
            result["converged"] = False

    return result


def run_column_generation(network_file: str, demand_file: str, time_limit: float, run_dir: Path) -> Dict[str, Any]:
    """Run Column Generation and parse output."""
    # Use persistent output directory
    output_dir = run_dir / "cg_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.column_generation.main",
        "--lmcf",
        "-i", network_file,
        "--demand-file", demand_file,
        "-o", str(output_dir),
        "--max-iter", "2000"
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 2,
            cwd=PROJECT_ROOT
        )
        elapsed = time.time() - start_time

        # Save console output
        save_console_output(run_dir, result.stdout, result.stderr)

        # Copy generated files to run directory
        if output_dir.exists():
            copy_generated_files(output_dir, run_dir)

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Exit code {result.returncode}",
                "time": elapsed,
                "run_directory": str(run_dir)
            }

        # Parse output
        parsed = parse_cg_output(result.stdout, elapsed)
        parsed["run_directory"] = str(run_dir)
        return parsed

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Timeout after {time_limit}s",
            "time": time_limit,
            "run_directory": str(run_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
            "run_directory": str(run_dir)
        }


def parse_cg_output(stdout: str, elapsed: float) -> Dict[str, Any]:
    """Parse Column Generation console output."""
    result = {"status": "unknown", "time": elapsed}

    # Extract objective
    obj_match = re.search(r'Objective value:\s*([0-9.]+)', stdout)
    if obj_match:
        result["objective"] = float(obj_match.group(1))

    # Extract iterations
    iter_match = re.search(r'Iterations:\s*(\d+)', stdout)
    if iter_match:
        result["iterations"] = int(iter_match.group(1))

    # Extract solve time
    time_match = re.search(r'Solve time:\s*([0-9.]+)', stdout)
    if time_match:
        result["time"] = float(time_match.group(1))

    # Extract convergence
    conv_match = re.search(r'Converged:\s*(Yes|No)', stdout)
    if conv_match:
        converged = conv_match.group(1) == "Yes"
        result["status"] = "success" if converged else "error"
        result["converged"] = converged

    return result


def run_dual_decomposition(network_file: str, demand_file: str, time_limit: float, run_dir: Path) -> Dict[str, Any]:
    """Run Dual Decomposition and parse output."""
    # Use persistent output directory
    output_dir = run_dir / "dual_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.dual.main",
        "--lmcf",
        "-i", network_file,
        "--demand-file", demand_file,
        "-o", str(output_dir)
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 2,
            cwd=PROJECT_ROOT
        )
        elapsed = time.time() - start_time

        # Save console output
        save_console_output(run_dir, result.stdout, result.stderr)

        # Copy generated files to run directory
        if output_dir.exists():
            copy_generated_files(output_dir, run_dir)

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Exit code {result.returncode}",
                "time": elapsed,
                "run_directory": str(run_dir)
            }

        # Parse summary.txt file
        summary_file = output_dir / "summary.txt"
        if summary_file.exists():
            parsed = parse_dual_summary(summary_file, elapsed)
        else:
            # Fallback to console parsing
            parsed = parse_dual_console(result.stdout, elapsed)

        parsed["run_directory"] = str(run_dir)
        return parsed

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Timeout after {time_limit}s",
            "time": time_limit,
            "run_directory": str(run_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
            "run_directory": str(run_dir)
        }


def parse_dual_summary(summary_file: Path, elapsed: float) -> Dict[str, Any]:
    """Parse Dual Decomposition summary.txt file."""
    result = {"status": "unknown", "time": elapsed}

    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse key-value pairs
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            try:
                if key == "time_sec":
                    result["time"] = float(value)
                elif key == "objective":
                    result["objective"] = float(value)
                elif key == "iterations":
                    result["iterations"] = int(value)
                elif key == "primal_residual":
                    result["primal_residual"] = float(value)
                elif key == "dual_residual":
                    result["dual_residual"] = float(value)
                elif key == "complementarity":
                    result["complementarity"] = float(value)
                elif key == "status":
                    is_optimal = "optimal" in value.lower()
                    result["status"] = "success" if is_optimal else "error"
                    result["converged"] = is_optimal
            except ValueError:
                continue

    return result


def parse_dual_console(stdout: str, elapsed: float) -> Dict[str, Any]:
    """Parse Dual Decomposition console output."""
    result = {"status": "unknown", "time": elapsed}

    # Extract objective
    obj_match = re.search(r'目标值.*?:\s*([0-9.]+)', stdout)
    if obj_match:
        result["objective"] = float(obj_match.group(1))

    # Extract iterations
    iter_match = re.search(r'迭代次数.*?:\s*(\d+)', stdout)
    if iter_match:
        result["iterations"] = int(iter_match.group(1))

    # Extract time
    time_match = re.search(r'求解时间.*?:\s*([0-9.]+)', stdout)
    if time_match:
        result["time"] = float(time_match.group(1))

    # Extract residuals
    primal_match = re.search(r'原始可行性残差.*?:\s*([0-9.e+-]+)', stdout)
    if primal_match:
        result["primal_residual"] = float(primal_match.group(1))

    dual_match = re.search(r'对偶可行性残差.*?:\s*([0-9.e+-]+)', stdout)
    if dual_match:
        result["dual_residual"] = float(dual_match.group(1))

    comp_match = re.search(r'互补性.*?:\s*([0-9.e+-]+)', stdout)
    if comp_match:
        result["complementarity"] = float(comp_match.group(1))

    # Check status (updated to match new format without Unicode symbols)
    if "收敛状态: 成功" in stdout or "optimal" in stdout.lower():
        result["status"] = "success"
        result["converged"] = True
    elif "收敛状态: 未成功" in stdout or "infeasible" in stdout.lower():
        result["status"] = "error"
        result["converged"] = False

    return result


def parse_interior_point_output(stdout: str, elapsed: float) -> Dict[str, Any]:
    """Parse Interior Point console output."""
    result = {"status": "unknown", "time": elapsed}

    # Extract objective value
    obj_match = re.search(r'目标值:\s*([0-9.]+)', stdout)
    if obj_match:
        result["objective"] = float(obj_match.group(1))

    # Extract solve time
    time_match = re.search(r'求解时间:\s*([0-9.]+)', stdout)
    if time_match:
        result["time"] = float(time_match.group(1))

    # Extract iterations
    iter_match = re.search(r'迭代次数:\s*(\d+)', stdout)
    if iter_match:
        result["iterations"] = int(iter_match.group(1))

    # Extract residuals
    primal_match = re.search(r'原始残差:\s*([0-9.e+-]+)', stdout)
    if primal_match:
        result["primal_residual"] = float(primal_match.group(1))

    dual_match = re.search(r'对偶残差:\s*([0-9.e+-]+)', stdout)
    if dual_match:
        result["dual_residual"] = float(dual_match.group(1))

    comp_match = re.search(r'互补性:\s*([0-9.e+-]+)', stdout)
    if comp_match:
        result["complementarity"] = float(comp_match.group(1))

    # Determine status (updated to match new format without Unicode symbols)
    if "收敛状态: 成功" in stdout or "solved" in stdout.lower():
        result["status"] = "success"
        result["converged"] = True
    elif "收敛状态: 未成功" in stdout or "error" in stdout.lower():
        result["status"] = "error"
        result["converged"] = False

    return result


def run_interior_point(network_file: str, demand_file: str, time_limit: float, run_dir: Path) -> Dict[str, Any]:
    """Run Interior Point method via CLI."""
    cmd = [
        sys.executable, "-m", "src.interior_point.main",
        "--network-file", network_file,
        "--demand-file", demand_file,
        "--time-limit-per-case", str(time_limit)
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 2,
            cwd=PROJECT_ROOT
        )
        elapsed = time.time() - start_time

        # Save console output
        save_console_output(run_dir, result.stdout, result.stderr)

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Exit code {result.returncode}",
                "time": elapsed,
                "run_directory": str(run_dir)
            }

        # Parse output
        parsed = parse_interior_point_output(result.stdout, elapsed)
        parsed["run_directory"] = str(run_dir)
        return parsed

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Timeout after {time_limit}s",
            "time": time_limit,
            "run_directory": str(run_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
            "run_directory": str(run_dir)
        }


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_single_benchmark(instance: Dict, implementation: str, run_number: int,
                        time_limit: float, output_dir: Path) -> Dict[str, Any]:
    """
    Run a single benchmark (one implementation on one instance).

    Args:
        instance: Instance metadata dict
        implementation: Implementation name
        run_number: Run number (1, 2, 3, ...)
        time_limit: Time limit in seconds
        output_dir: Base output directory for all results

    Returns:
        Result dictionary with metrics
    """
    network_file = instance["network_file"]
    demand_file = instance["demand_file"]

    print(f"    Run {run_number}: {implementation}...", end=" ", flush=True)

    # Create run directory
    run_dir = create_run_directory(
        output_dir=output_dir,
        instance_name=instance["name"],
        implementation=implementation,
        run_number=run_number
    )

    start = time.time()

    if implementation == "simplex":
        result = run_simplex(network_file, demand_file, time_limit, run_dir)
    elif implementation == "column_generation":
        result = run_column_generation(network_file, demand_file, time_limit, run_dir)
    elif implementation == "dual":
        result = run_dual_decomposition(network_file, demand_file, time_limit, run_dir)
    elif implementation == "interior_point":
        result = run_interior_point(network_file, demand_file, time_limit, run_dir)
    else:
        result = {"status": "error", "error": f"Unknown implementation: {implementation}"}

    elapsed = time.time() - start

    # Add metadata
    result.update({
        "instance_name": instance["name"],
        "category": instance["category"],
        "implementation": implementation,
        "run_number": run_number,
        "total_elapsed": elapsed
    })

    # Print result
    status_symbol = "✓" if result.get("converged", False) else "✗"
    time_str = f"{result.get('time', 0):.2f}s"
    obj_str = f"obj={result.get('objective', 0):.2f}" if 'objective' in result else "N/A"
    print(f"{status_symbol} {time_str} {obj_str}")

    return result


def run_full_benchmark(instances: List[Dict], implementations: List[str],
                      num_runs: int, time_limit: float,
                      output_dir: Path,
                      checkpoint_file: Optional[Path] = None) -> List[Dict]:
    """
    Run full benchmark on all instances and implementations.

    Args:
        instances: List of instance dicts
        implementations: List of implementation names to test
        num_runs: Number of runs per instance+implementation
        time_limit: Time limit per run
        output_dir: Output directory for results
        checkpoint_file: Optional checkpoint file for resuming

    Returns:
        List of all result dictionaries
    """
    results = []
    total_runs = len(instances) * len(implementations) * num_runs
    completed = 0

    # Load checkpoint if exists
    completed_set = set()
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            completed_set = set(checkpoint.get("completed", []))
            completed = len(completed_set)
            print(f"Resuming from checkpoint: {completed}/{total_runs} runs completed")

    start_time = time.time()

    for inst_idx, instance in enumerate(instances, 1):
        print(f"\n[{inst_idx}/{len(instances)}] Instance: {instance['name']}")
        print(f"  Network: {instance['network_name']}, Demand: {instance['demand_name']}")

        for impl in implementations:
            for run_num in range(1, num_runs + 1):
                # Check if already completed
                task_id = f"{instance['name']}_{impl}_{run_num}"
                if task_id in completed_set:
                    completed += 1
                    continue

                # Run benchmark
                result = run_single_benchmark(instance, impl, run_num, time_limit, output_dir)
                results.append(result)
                completed_set.add(task_id)
                completed += 1

                # Update progress
                elapsed = time.time() - start_time
                avg_time_per_run = elapsed / completed if completed > 0 else 0
                eta = avg_time_per_run * (total_runs - completed)

                print(f"  Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%) "
                      f"| ETA: {eta/60:.1f} min")

                # Save checkpoint every 10 runs
                if checkpoint_file and completed % 10 == 0:
                    save_checkpoint(checkpoint_file, results, list(completed_set))

    return results


def save_checkpoint(checkpoint_file: Path, results: List[Dict], completed: List[str]):
    """Save checkpoint data."""
    checkpoint = {
        "results": results,
        "completed": completed,
        "timestamp": time.time()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(runs: List[Dict]) -> Dict[str, Any]:
    """
    Compute statistics for multiple runs of the same instance+implementation.

    Args:
        runs: List of result dicts for the same instance+implementation

    Returns:
        Dictionary with aggregated statistics
    """
    if not runs:
        return {}

    # Count successes
    successes = [r for r in runs if r.get("status") == "success"]
    success_rate = len(successes) / len(runs) if runs else 0

    stats = {
        "num_runs": len(runs),
        "success_count": len(successes),
        "success_rate": success_rate,
    }

    if not successes:
        return stats

    # Extract metrics from successful runs
    times = [r.get("time", 0) for r in successes]
    objectives = [r.get("objective") for r in successes if "objective" in r]
    iterations = [r.get("iterations") for r in successes if "iterations" in r]
    primal_residuals = [r.get("primal_residual") for r in successes if "primal_residual" in r]
    dual_residuals = [r.get("dual_residual") for r in successes if "dual_residual" in r]
    complementarities = [r.get("complementarity") for r in successes if "complementarity" in r]

    # Compute statistics
    if times:
        stats.update({
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": min(times),
            "max_time": max(times),
        })

    if objectives:
        stats.update({
            "avg_objective": np.mean(objectives),
            "std_objective": np.std(objectives),
            "min_objective": min(objectives),
            "max_objective": max(objectives),
        })

    if iterations:
        stats.update({
            "avg_iterations": np.mean(iterations),
            "std_iterations": np.std(iterations),
            "min_iterations": min(iterations),
            "max_iterations": max(iterations),
        })

    if primal_residuals:
        stats.update({
            "avg_primal_residual": np.mean(primal_residuals),
            "std_primal_residual": np.std(primal_residuals),
        })

    if dual_residuals:
        stats.update({
            "avg_dual_residual": np.mean(dual_residuals),
            "std_dual_residual": np.std(dual_residuals),
        })

    if complementarities:
        stats.update({
            "avg_complementarity": np.mean(complementarities),
            "std_complementarity": np.std(complementarities),
        })

    return stats


def aggregate_results(results: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Aggregate results by instance+implementation.

    Args:
        results: List of all result dicts

    Returns:
        Dictionary mapping (instance_name, implementation) to statistics
    """
    # Group results by (instance, implementation)
    grouped = defaultdict(list)
    for result in results:
        key = (result["instance_name"], result["implementation"])
        grouped[key].append(result)

    # Compute statistics for each group
    aggregated = {}
    for key, runs in grouped.items():
        stats = compute_statistics(runs)
        stats["instance_name"] = key[0]
        stats["implementation"] = key[1]
        stats["category"] = runs[0]["category"]
        aggregated[key] = stats

    return aggregated


# ============================================================================
# Output Generation
# ============================================================================

def save_detailed_results(results: List[Dict], output_file: Path):
    """Save detailed results (all runs) to CSV."""
    if not results:
        return

    fieldnames = [
        "instance_name", "category", "implementation", "run_number",
        "status", "converged", "time", "objective", "iterations",
        "primal_residual", "dual_residual", "complementarity",
        "error", "total_elapsed", "run_directory"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\n✓ Detailed results saved to: {output_file}")


def save_summary_results(aggregated: Dict, output_file: Path):
    """Save aggregated summary results to CSV."""
    if not aggregated:
        return

    fieldnames = [
        "instance_name", "category", "implementation",
        "num_runs", "success_count", "success_rate",
        "avg_time", "std_time", "min_time", "max_time",
        "avg_objective", "std_objective",
        "avg_iterations", "std_iterations",
        "avg_primal_residual", "std_primal_residual",
        "avg_dual_residual", "std_dual_residual",
        "avg_complementarity", "std_complementarity"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for stats in sorted(aggregated.values(), key=lambda x: (x["category"], x["instance_name"], x["implementation"])):
            writer.writerow(stats)

    print(f"✓ Summary results saved to: {output_file}")


def print_summary(aggregated: Dict, implementations: List[str]):
    """Print summary statistics to console."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)

    for impl in implementations:
        impl_results = [v for k, v in aggregated.items() if k[1] == impl]
        if not impl_results:
            continue

        total_instances = len(impl_results)
        total_runs = sum(r["num_runs"] for r in impl_results)
        successful = sum(r["success_count"] for r in impl_results)
        success_rate = successful / total_runs if total_runs > 0 else 0

        avg_times = [r["avg_time"] for r in impl_results if "avg_time" in r]
        overall_avg_time = sum(avg_times) / len(avg_times) if avg_times else 0

        print(f"\n{impl.upper()}")
        print(f"  Instances tested: {total_instances}")
        print(f"  Total runs: {total_runs}")
        print(f"  Successful runs: {successful}/{total_runs} ({success_rate*100:.1f}%)")
        print(f"  Average time: {overall_avg_time:.2f}s")

        # Add quality metrics
        avg_primal = [r.get("avg_primal_residual") for r in impl_results if "avg_primal_residual" in r]
        if avg_primal:
            print(f"  Avg primal residual: {np.mean(avg_primal):.2e}")

        avg_dual = [r.get("avg_dual_residual") for r in impl_results if "avg_dual_residual" in r]
        if avg_dual:
            print(f"  Avg dual residual: {np.mean(avg_dual):.2e}")

        avg_comp = [r.get("avg_complementarity") for r in impl_results if "avg_complementarity" in r]
        if avg_comp:
            print(f"  Avg complementarity: {np.mean(avg_comp):.2e}")

    print("\n" + "="*100)


# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive LMCF Benchmark for All Implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=600,
        help="Time limit per run (seconds)"
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per instance+implementation"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results"
    )

    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        default=IMPLEMENTATIONS,
        help="Implementations to test"
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=CATEGORIES,
        default=None,
        help="Test only instances from this category"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Discover instances
    print("="*100)
    print("LMCF COMPREHENSIVE BENCHMARK")
    print("="*100)
    print(f"\nDiscovering LMCF instances...")
    instances = discover_lmcf_instances(args.category)

    if not instances:
        print("Error: No instances found!")
        return 1

    print(f"Found {len(instances)} instances")
    for category in CATEGORIES:
        cat_instances = [i for i in instances if i["category"] == category]
        if cat_instances:
            print(f"  {category}: {len(cat_instances)}")

    # Configuration summary
    total_runs = len(instances) * len(args.implementations) * args.num_runs
    estimated_time = total_runs * args.time_limit / 60  # Rough estimate

    print(f"\nConfiguration:")
    print(f"  Implementations: {', '.join(args.implementations)}")
    print(f"  Runs per instance: {args.num_runs}")
    print(f"  Time limit per run: {args.time_limit}s")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated max time: {estimated_time:.1f} minutes")
    print(f"  Output directory: {args.output}")

    # Checkpoint file
    checkpoint_file = args.output / "benchmark_progress.json" if args.resume else None

    input("\nPress Enter to start benchmark...")

    # Run benchmark
    start_time = time.time()
    results = run_full_benchmark(
        instances,
        args.implementations,
        args.num_runs,
        args.time_limit,
        args.output,
        checkpoint_file
    )
    total_time = time.time() - start_time

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(results)

    # Save outputs
    print("\nSaving results...")
    save_detailed_results(results, args.output / "detailed_results.csv")
    save_summary_results(aggregated, args.output / "summary_results.csv")

    # Print summary
    print_summary(aggregated, args.implementations)

    print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")
    print(f"\nAll results saved to: {args.output}")

    # Clean up checkpoint
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
