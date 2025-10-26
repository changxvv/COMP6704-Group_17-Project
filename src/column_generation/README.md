# Column Generation for Multi-Commodity Network Flow

Modern Python implementation of Dantzig-Wolfe decomposition for solving Multi-Commodity Network Flow (MCNF) problems using Gurobi.

## Features

- **Multi-Format Support**: Parses both SNDlib and LMCF format network files
- **Modern Python**: Type hints, logging, pathlib, f-strings, PEP 8 compliant
- **Gurobi Integration**: High-performance commercial solver with academic license support
- **Modular Design**: Clean separation of concerns across multiple modules
- **Comprehensive Logging**: Detailed iteration tracking and debugging information
- **Automated Benchmarking**: Built-in benchmark runner for testing multiple datasets

## Installation

```bash
# Install dependencies
pip install gurobipy

# Gurobi requires a license (free academic licenses available)
# Visit: https://www.gurobi.com/academia/academic-program-and-licenses/
```

## Usage

### SNDlib Format

```bash
# Run from project root
python -m src.column_generation.main -i tests/ta1.txt -o output/ta1

# With options
python -m src.column_generation.main \
    --input tests/ta1.txt \
    --output output/ta1 \
    --epsilon 1e-6 \
    --max-iter 2000 \
    --verbose

# For directed networks (default is undirected)
python -m src.column_generation.main \
    --input tests/SNDlib/janos-us.txt \
    --directed
```

### LMCF Format

```bash
# LMCF format uses separate network and demand files
python -m src.column_generation.main \
    --lmcf \
    -i tests/LMCF/OtherDemands/C22.txt \
    --demand-file tests/LMCF/OtherDemands/D22.txt \
    -o output/lmcf

# All LMCF networks are undirected (default behavior)
```

### Benchmarking

```bash
# Run all SNDlib benchmarks
python -m src.column_generation.benchmark \
    --dataset sndlib \
    --timeout 300

# Run all LMCF benchmarks
python -m src.column_generation.benchmark \
    --dataset lmcf \
    --timeout 600

# Filter by pattern (e.g., only Grid problems)
python -m src.column_generation.benchmark \
    --dataset lmcf \
    --filter "gd*"

# Auto-detect format from directory
python -m src.column_generation.benchmark \
    --networks-dir tests/LMCF
```

## Module Structure

### `parser.py`
Parses SNDlib format files containing:
- NODES section: Node IDs with coordinates (supports negative coordinates)
- LINKS section: Directed edges with capacity modules and costs
- DEMANDS section: Commodity demands with UNLIMITED max path length support
- **Capacity Strategy**: Prioritizes module capacity over pre-installed capacity

### `lmcf_parser.py`
Parses LMCF format files (NEW):
- Network file (C*.txt): startnode, endnode, cost, capacity
- Demand file (D*.txt): origin, destination, demand
- Simple tabular format with integer node IDs
- Returns same data structure as SNDlib parser for compatibility

### `data_models.py`
Core data structures:
- `Node`: Network node with successors/predecessors
- `Edge`: Directed edge with capacity and cost
- `Demand`: Commodity flow requirement
- `Network`: Complete network graph with dual format support
  - `load_from_sndlib()`: Load SNDlib format
  - `load_from_lmcf()`: Load LMCF format

### `pricing_problem.py`
Pricing subproblem solver:
- `dijkstra()`: Shortest path algorithm with modified weights
- `generate_initial_solution()`: Initial feasible solution
- `solve_pricing_problem()`: Find columns with negative reduced cost

### `master_problem.py`
Restricted Master Problem:
- `MasterProblem`: RMP solver using Gurobi
- Dynamic column addition (Gurobi Column objects)
- Efficient dual variable extraction
- Big-M method for artificial variables

### `column_generation.py`
Main algorithm controller:
- `ColumnGeneration`: Orchestrates DW decomposition
- Iterative RMP-pricing loop
- Convergence checking and result extraction
- Configurable Big-M penalty parameter

### `utils.py`
Helper functions:
- Network state reset operations
- Solution validation
- Objective value computation
- Capacity-aware edge detection

### `benchmark.py`
Automated benchmark runner (NEW):
- Support for both SNDlib and LMCF datasets
- Automatic file pairing for LMCF format
- CSV output with performance metrics
- Pattern filtering and timeout support
- Summary statistics and rankings

### `main.py`
Command-line interface and entry point with dual format support

## Output Files

The algorithm generates four output files:

1. **iterations.txt**: Iteration history
   - Iteration number, objective value, reduced cost, number of columns

2. **variables.txt**: Non-zero lambda values
   - Variable name and value for each column used in final solution

3. **routes.txt**: Demand routing
   - Demand ID, flow ratio, route as node sequence

4. **flow.txt**: Edge flows
   - Edge ID, source, target, flow, capacity

## Algorithm Details

The implementation uses **Dantzig-Wolfe decomposition**:

1. **Master Problem (RMP)**: Convex combination of extreme points
   - Variables: λ_k (weight for each column/extreme point)
   - Objective: min Σ(cost_k × λ_k)
   - Constraints: Capacity constraints + convexity (Σλ = 1)

2. **Pricing Subproblem**: Find improving columns
   - Modify edge weights using dual variables
   - Run Dijkstra for each demand
   - Check reduced cost for improvement

3. **Iteration**: Repeat until no improving columns found

## Supported Datasets

### SNDlib Networks (tests/SNDlib/)
Industry-standard telecommunications network library:
- 27 networks ranging from small (abilene: 12 nodes) to large (germany50: 50 nodes)
- 4 directed networks: GIUL39, JANOS-US, JANOS-US-CA, SUN
- 23 undirected networks (automatic bidirectional edge creation)

### LMCF Networks (tests/LMCF/)
Academic benchmark suite for Multi-Commodity Flow problems:
- **34 problem instances** across 4 categories
- **Planar**: 10 problems (30-2500 nodes, grid-like topology)
- **Grid**: 15 problems (25-1225 nodes, regular grid structure)
- **Traffic**: 6 problems (24-13,389 nodes, real road networks)
- **Other**: 3 problems (ndo22, ndo148, 904 - challenging instances)

All LMCF networks are undirected.

**Important:** The LMCF files in this repository have been corrected for format inconsistencies. The original dataset had swapped cost/capacity columns in Planar/Grid/Traffic categories. Files have been fixed to match the documented format. See `.agent/tasks/lmcf_format_fix.md` for details.

## Known Limitations and Design Decisions

1. **Undirected Networks**: By default, the implementation treats networks as undirected by creating reverse edges. Use `--directed` flag only for networks that already have bidirectional links in the input.

2. **SNDlib Capacity Strategy**: The parser prioritizes module capacity over pre-installed capacity:
   - Priority 1: Largest capacity module (typically 4-50× larger than pre-installed)
   - Priority 2: Pre-installed capacity
   - Priority 3: Uncapacitated (infinite capacity)
   - Unit cost uses amortized module cost (module_cost / module_capacity)

3. **Infeasible Networks**: Some SNDlib networks may be infeasible due to tight capacity constraints. These are network design problems where capacity installation is required, not pure routing problems.

4. **Big-M Parameter**: Artificial variables use Big-M penalty (default: 1e9). Adjust with `--big-m` if convergence issues occur.

5. **LMCF Dataset Format**: The original LMCF dataset had inconsistent column ordering (cost/capacity swapped in 31 of 34 files). This repository contains corrected files. If using external LMCF sources, verify column order matches expected value ranges (small capacities < 1000, large costs).

6. **Gurobi License**: Full academic license required for larger LMCF problems. Restricted/trial licenses have hidden limitations that prevent solving problems with >~300 variables or iterative solving.

## Performance Highlights

### Gurobi Migration (11.8× Speedup)
Migration from PuLP to Gurobi achieved significant performance improvements:
- **ta1**: 1.77s → 0.15s (11.8× faster)
- **ta2**: 4.35s → 0.37s (11.8× faster)
- Dynamic column addition (Gurobi Column objects)
- Efficient dual variable extraction

### LMCF Validation
Successfully validated against published results:
- **ndo22**: 1882.38 vs expected 1882.37 (0.0005% error)
- **Planar 30**: 44,350,624 vs expected 44,350,800 (0.0004% error)
- Converges in ~90 iterations for medium problems
- Format correction applied to 31 of 34 files (see Known Limitations)

## Comparison with Reference Implementation

This implementation refactors `refs/Multi-Commodity-Network-Flow/main.py` with:

- ✅ Modern Python style (Python 3.10+)
- ✅ Dual format support (SNDlib + LMCF)
- ✅ Gurobi with dynamic column addition
- ✅ Modular file structure (9 files vs 1 monolithic file)
- ✅ Type hints throughout
- ✅ Comprehensive logging
- ✅ Command-line interface
- ✅ Automated benchmarking system

## Example

### SNDlib Format

```python
from pathlib import Path
from src.column_generation import Network, ColumnGeneration

# Load SNDlib network
network = Network()
network.load_from_sndlib(Path("tests/ta1.txt"), undirected=True)

# Create solver
cg = ColumnGeneration(
    network=network,
    epsilon=1e-6,
    max_iterations=2000,
    output_folder="./output/",
    M=1e9  # Big-M penalty for artificial variables
)

# Run algorithm
results = cg.run()

# Retrieve and save solution
cg.retrieve_solution()
cg.save_results()

print(f"Objective: {results['objective']:.2f}")
print(f"Iterations: {results['iterations']}")
print(f"Converged: {results['converged']}")
```

### LMCF Format

```python
from pathlib import Path
from src.column_generation import Network, ColumnGeneration

# Load LMCF network (separate files)
network = Network()
network.load_from_lmcf(
    network_file=Path("tests/LMCF/OtherDemands/C22.txt"),
    demand_file=Path("tests/LMCF/OtherDemands/D22.txt"),
    undirected=True  # All LMCF networks are undirected
)

# Create and run solver (same as SNDlib)
cg = ColumnGeneration(network=network, output_folder="./output/")
results = cg.run()
cg.retrieve_solution()
cg.save_results()
```

## References

- **Dantzig-Wolfe Decomposition**: Column Generation technique for large-scale optimization
- **Multi-Commodity Network Flow**: Simultaneous routing of multiple commodities
- **SNDlib**: Survivable Network Design Library - http://sndlib.zib.de/
- **LMCF Dataset**: Babonneau et al. (2005) - Capacited Linear Multicommodity Flow Problems
- **Gurobi Optimization**: https://www.gurobi.com/

## License

See project root LICENSE file.
