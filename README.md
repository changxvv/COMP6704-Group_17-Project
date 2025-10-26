# COMP 6704 Group Project: Supply Chain Network Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)]()

**Multi-Commodity Network Flow (MCNF) Optimization**

A comprehensive implementation of four optimization algorithms for solving the capacitated multi-commodity network flow problem, with extensive benchmarking and visualization capabilities.

---

## Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Problem Definition](#problem-definition)
- [Mathematical Formulation](#mathematical-formulation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running Individual Algorithms](#running-individual-algorithms)
  - [Running Benchmarks](#running-benchmarks)
  - [Generating Visualizations](#generating-visualizations)
- [Algorithm Implementations](#algorithm-implementations)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [License](#license)

---

## About

This project implements and compares **four optimization algorithms** for solving the **Multi-Commodity Network Flow (MCNF)** problem in supply chain network optimization. The goal is to route multiple commodities through a shared network while minimizing total transportation cost, subject to capacity constraints.

Developed as part of the COMP 6704 course (Optimization Algorithms), this project includes:
- Four complete algorithm implementations
- Comprehensive benchmarking infrastructure (240 test executions)
- Publication-quality visualizations
- Extensive documentation

---

## Key Features

✅ **Four Algorithm Implementations**
- Simplex Method (Primal/Dual/Revised variants)
- Column Generation (Dantzig-Wolfe decomposition)
- Dual Decomposition (Lagrangian relaxation with subgradient method)
- Interior Point Method (Primal-dual barrier method)

✅ **Comprehensive Benchmarking**
- Automated benchmark runner
- 20 LMCF instances across 4 categories
- 240 total test executions (20 instances × 4 algorithms × 3 runs)
- Statistical analysis with multi-run aggregation

✅ **Publication-Quality Visualization**
- 5 professional plots (300 DPI)
- Success rate, computation time, solution precision analysis
- Ground truth comparison with best known objectives

✅ **Complete Documentation**
- Detailed algorithm descriptions
- Reproducibility guides
- Troubleshooting resources

---

## Problem Definition

Route multiple commodities through a directed network composed of factories/warehouses/customers, delivering each commodity's demand from supply sources to demand destinations.

Transportation costs are linear (unit cost × flow), allowing each commodity to be split across multiple paths. Each transportation link has a shared capacity constraint—the total flow of all commodities on that link cannot exceed its capacity. All commodity demands must be satisfied, and warehouses serve only as transshipment nodes (intermediate nodes must conserve flow for each commodity).

**Objective**: Minimize total transportation cost.

Specifically, this is a **minimum-cost multi-commodity network flow** problem with linear costs, splittable flows (continuous variables), and shared arc capacities.

---

## Mathematical Formulation

### Arc-Based Minimum-Cost MCNF (Standard LP Formulation)

Given a directed graph $G=(V,A)$ where $V$ is the set of nodes (warehouses) and $A$ is the set of arcs (transportation links), and a set of commodities $K$. Each arc $(i,j)\in A$ has a shared capacity $u_{ij}\ge0$. The unit transportation cost for commodity $k$ on arc $(i,j)$ is $c^{k}_{ij}$.

Let $b_i^k$ denote the net supply of node $i$ for commodity $k$ (positive for sources, negative for sinks, zero for transshipment nodes, with $\sum_i b_i^k=0$). The decision variable $x^{k}_{ij}\ge0$ represents the flow of commodity $k$ on arc $(i,j)$:

$$
\begin{aligned}
\min \ & \sum_{k\in K}\sum_{(i,j)\in A} c^{k}_{ij}\,x^{k}_{ij}\\
\text{s.t. }
& \sum_{(i,j)\in A} x^{k}_{ij}-\sum_{(j,i)\in A} x^{k}_{ji}=b_i^k, &&\forall i\in V,\ \forall k\in K \quad(\text{Flow Balance})\\
& \sum_{k\in K} x^{k}_{ij}\le u_{ij}, &&\forall (i,j)\in A \quad(\text{Shared Capacity})\\
& x^{k}_{ij}\ge0, &&\forall (i,j)\in A,\ \forall k\in K .
\end{aligned}
$$

Equivalently, each commodity can be specified by $(s_k,t_k,d_k)$ denoting its source, sink, and demand (where $b_{s_k}^k=d_k$, $b_{t_k}^k=-d_k$, and all other nodes have $b_i^k=0$).

---

## Prerequisites

### Required Software
- **uv**: refer to [uv installation](https://github.com/astral-sh/uv#installation) to install uv and set up the `PATH` environment variable as prompted.
- **Gurobi Optimizer**: Version 12.0.3 or higher
  - Academic license required (free for students/faculty)
  - Download and license instructions: [Gurobi Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/)
  - Required for Column Generation algorithm
  - Optional for other algorithms (fallback to open-source solvers available)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd COMP6704-Group_17-Project/
```

### 2. Install Dependencies

```bash
# Install project with all dependencies
uv sync
```

This will install all required packages:
- `gurobipy>=12.0.3` - Gurobi Python API
- `networkx>=3.4.2` - Network algorithms
- `numpy>=2.2.6` - Numerical computing
- `scipy>=1.15.3` - Scientific computing
- `pulp>=3.3.0` - LP modeling
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data analysis

### 3. Configure Gurobi License (Optional)

After installing Gurobi, obtain and activate your academic license:

```bash
# Download your license file from Gurobi website
# Place gurobi.lic in your home directory or set GRB_LICENSE_FILE environment variable

# Verify Gurobi installation
gurobi.sh  # On Linux/macOS
gurobi.bat  # On Windows
```

For detailed Gurobi setup instructions, visit: https://www.gurobi.com/documentation/

---

## Quick Start

### Run Full Benchmark Suite

```bash
# Run all algorithms on all 20 instances (3 runs each = 240 executions)
uv run benchmark_all_lmcf.py --time-limit 180 --num-runs 3
```

### Run a Single Algorithm

```bash
# Simplex method on LMCF dataset
uv run -m src.simplex.main --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/
```

### Generate Visualizations

```bash
# Create publication-quality plots from benchmark results
uv run plot_benchmark_results.py
```

Results will be saved in `benchmark_results/figures/`.

---

## Usage

### Running Individual Algorithms

All algorithms support both SNDlib and LMCF dataset formats.

#### Simplex Method

```bash
# Dense variant (for small-medium problems)
uv run -m src.simplex.main --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/simplex/

# Sparse variant (for larger problems)
uv run -m src.simplex.main_sparse --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/simplex/
```

#### Column Generation

```bash
uv run -m src.column_generation.main --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/column_gen/
```

#### Dual Decomposition

```bash
uv run -m src.dual.main --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/dual/
```

#### Interior Point Method

```bash
uv run -m src.interior_point.main --lmcf \
  -i tests/LMCF/GridDemands/Cgd1.txt \
  --demand-file tests/LMCF/GridDemands/Dgd1.txt \
  -o output/interior/
```

### Running Benchmarks

The benchmark suite supports various configurations:

```bash
# Full benchmark (all 20 instances, 4 algorithms, 3 runs each)
uv run benchmark_all_lmcf.py --time-limit 180 --num-runs 3

# Test specific category
uv run benchmark_all_lmcf.py --category GridDemands --num-runs 1

# Test specific algorithms
uv run benchmark_all_lmcf.py --implementations simplex column_generation

# Quick test (single run)
uv run benchmark_all_lmcf.py --category GridDemands --num-runs 1 --time-limit 60
```

Results are saved in:
- `benchmark_results/detailed_results.csv` - All individual runs
- `benchmark_results/summary_results.csv` - Aggregated statistics
- `benchmark_results/runs/` - Per-run artifacts (stdout, stderr, files)

### Generating Visualizations

```bash
# Generate all 5 publication-quality plots
uv run plot_benchmark_results.py
```

Outputs 5 PNG files (300 DPI) in `benchmark_results/figures/`:
1. `01_overall_success_rate.png` - Algorithm reliability
2. `02_average_time.png` - Computational efficiency
3. `03_time_distribution.png` - Performance variability (violin plots)
4. `04_solution_precision.png` - Solution quality vs ground truth
5. `05_iteration_distribution.png` - Algorithmic efficiency

---

## Algorithm Implementations

All four algorithms are fully implemented with detailed documentation in their respective directories.

### Simplex Method (Primal/Dual Simplex)

**Location**: `src/simplex/`

**Approach**:
- Moves along edges of the feasible region, selecting non-basic variables that most reduce the objective
- **Primal Simplex**: Maintains primal feasibility throughout
- **Dual Simplex**: Maintains dual feasibility while gradually restoring primal feasibility (more stable when starting with an infeasible basis)
- **Revised Simplex**: Memory-efficient variant using basis factorization

**Strengths**:
- Exact optimal solutions
- Well-suited for small to medium problems
- Robust and proven

**Considerations**:
- Dense matrix operations can be slow for large problems
- May time out on instances with >500 nodes

**Documentation**: [src/simplex/README.md](src/simplex/README.md)

---

### Interior-Point Method (Barrier Method / Predictor-Corrector)

**Location**: `src/interior_point/`

**Approach**:
- Approaches the optimal boundary from the interior of the feasible region along a **barrier trajectory**
- Uses the Mehrotra Predictor-Corrector method
- Each iteration solves KKT linear systems (sparse symmetric indefinite or positive definite after regularization)
- Typically converges within tens of iterations

**Strengths**:
- Polynomial-time complexity
- Predictable iteration count
- Good for large dense problems

**Considerations**:
- Numerical stability issues on ill-conditioned problems
- Requires careful parameter tuning

**Documentation**: [src/interior_point/README.md](src/interior_point/README.md)

---

### Column Generation (Dantzig-Wolfe Decomposition)

**Location**: `src/column_generation/`

**Approach**:
- Reformulates each commodity's flow using **path variables** (feasible paths from source to sink)
- Capacity constraints become coupling constraints
- Solves a **Restricted Master Problem (RMP)** containing only a small subset of currently selected paths
- **Pricing Subproblem**: For each commodity, solves a **shortest path problem** with arc weights as **reduced costs** (original cost minus dual prices from capacity constraints)
- If negative reduced cost paths are found, adds them as new columns to the RMP
- Iterates until no improving columns exist

**Strengths**:
- Exact optimal solutions
- Handles medium to large problems efficiently
- Strong theoretical bounds
- Most robust across diverse instances (60-75% success rate)

**Considerations**:
- Requires Gurobi license
- Can be slower on very small instances

**Documentation**: [src/column_generation/README.md](src/column_generation/README.md)

---

### Dual Decomposition (Lagrangian Relaxation with Subgradient Method)

**Location**: `src/dual/`

**Approach**:
- Applies Lagrangian relaxation to the shared capacity constraints
- Decomposes the problem by commodity using dual prices (Lagrange multipliers)
- Each commodity solves an independent **shortest path subproblem** with modified arc costs (original cost + dual price)
- Updates dual variables using the **subgradient method** based on capacity violations
- Iterates until convergence or maximum iterations reached

**Strengths**:
- Highly parallelizable (commodities solved independently)
- Scalable to very large problems
- Memory efficient
- Good for problems with many commodities

**Considerations**:
- Convergence depends on step size tuning
- May require many iterations (up to 1000+)
- Solution quality depends on convergence tolerance
- Does not guarantee primal feasibility (may violate capacity constraints slightly)

**Documentation**: [src/dual/README.md](src/dual/README.md)

---

## Project Structure

```
COMP6704-group_project/
├── src/                          # Algorithm implementations
│   ├── simplex/                  # Simplex method (dense & sparse variants)
│   │   ├── main.py               # Main entry point
│   │   ├── main_sparse.py        # Sparse variant
│   │   ├── simplex_solver.py     # Core solver logic
│   │   ├── parser.py             # Input parsers
│   │   └── README.md             # Algorithm documentation
│   ├── column_generation/        # Column generation (Dantzig-Wolfe)
│   │   ├── main.py
│   │   ├── column_generation.py  # Main algorithm
│   │   ├── master_problem.py     # RMP solver
│   │   ├── pricing_problem.py    # Shortest path subproblems
│   │   └── README.md
│   ├── dual/                     # Dual decomposition
│   │   ├── main.py
│   │   ├── dual_decomp.py        # Lagrangian relaxation
│   │   ├── dual_solver.py        # Subgradient method
│   │   └── README.md
│   └── interior_point/           # Interior point method
│       ├── main.py
│       ├── lib/
│       │   ├── solvers.py        # IPM solver
│       │   └── modeling.py       # LP formulation
│       └── README.md
├── tests/                        # Benchmark datasets
│   ├── LMCF/                     # LMCF benchmarks (34 instances)
│   │   ├── GridDemands/          # 9 grid network instances
│   │   ├── PlanarNetworks/       # 5 planar graph instances
│   │   ├── TrafficNetworks/      # 3 traffic network instances
│   │   ├── OtherDemands/         # 3 diverse instances
│   │   ├── LMCF_Instances.md     # Dataset documentation
│   │   └── format.txt            # File format specification
│   └── SNDlib/                   # SNDlib benchmarks (27 instances)
├── benchmark_all_lmcf.py         # Comprehensive benchmark runner (1,074 lines)
├── plot_benchmark_results.py     # Visualization suite (420 lines)
├── benchmark_results/            # Benchmark outputs
│   ├── detailed_results.csv      # All 240 individual runs
│   ├── summary_results.csv       # Aggregated statistics
│   ├── runs/                     # Per-run artifacts (stdout, stderr, files)
│   └── figures/                  # 5 visualization plots (300 DPI PNG)
├── pyproject.toml                # Python project configuration
└── README.md                     # This file
```

---

## Datasets

### LMCF Benchmarks

**Source**: Babonneau et al. (2005) - Capacitated Linear Multicommodity Flow Problems

**Location**: `tests/LMCF/`

**Format**: Separate network (C*.txt) and demand (D*.txt) files
- Network file: `startnode endnode cost capacity`
- Demand file: `source target demand`

**Categories**:
- **GridDemands** (9 instances): Grid networks of varying sizes
- **PlanarNetworks** (5 instances): Planar graphs (30-150 nodes)
- **TrafficNetworks** (3 instances): Real traffic networks (Chicago, Sioux Falls, Winnipeg)
- **OtherDemands** (3 instances): Diverse network topologies

**Note**: This repository contains corrected LMCF files (files had swapped cost/capacity columns in the original distribution).

**Documentation**: [tests/LMCF/LMCF_Instances.md](tests/LMCF/LMCF_Instances.md)

---

## Benchmarking

### Running Benchmarks

The `benchmark_all_lmcf.py` script provides comprehensive automated benchmarking:

```bash
# Full benchmark (recommended)
uv run benchmark_all_lmcf.py --time-limit 180 --num-runs 3
```

**Configuration Options**:
- `--time-limit`: Timeout per run in seconds (default: 180)
- `--num-runs`: Number of runs per (instance, algorithm) pair (default: 3)
- `--category`: Test specific category only (GridDemands, PlanarNetworks, etc.)
- `--implementations`: Test specific algorithms only (space-separated)

**Benchmark Scope**:
- **20 LMCF instances** across 4 categories
- **4 algorithm implementations**
- **3 runs each** for statistical reliability
- **240 total executions**

### Understanding Results

**Output Files**:
1. `benchmark_results/detailed_results.csv` - All 240 individual runs with metrics
2. `benchmark_results/summary_results.csv` - Aggregated statistics (mean, std, min, max)
3. `benchmark_results/runs/*/` - Per-run artifacts (stdout, stderr, generated files)

**Key Metrics**:
- **success_rate**: Percentage of runs that converged successfully
- **avg_time**: Average computation time (successful runs only)
- **avg_objective**: Average objective value achieved
- **avg_iterations**: Average number of iterations
- **primal_residual**: Primal feasibility measure
- **dual_residual**: Dual feasibility measure
- **complementarity**: Optimality gap measure

---

## Results

### Benchmark Visualizations

After running `plot_benchmark_results.py`, five publication-quality plots are generated in `benchmark_results/figures/`:

1. **Overall Success Rate** - Shows reliability across all 20 instances
2. **Average Computation Time** - Performance on successful runs (with error bars)
3. **Time Distribution** - Violin plots showing performance variability
4. **Solution Precision** - Relative error vs best known objectives
5. **Iteration Distribution** - Algorithmic efficiency (independent of implementation speed)

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 COMP6704 Group Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
