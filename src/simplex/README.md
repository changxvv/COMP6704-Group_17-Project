Run LMCF:

```
python -m src.simplex.main tests/LMCF/GridDemands/Cgd1.txt --format lmcf --demand-file tests/LMCF/GridDemands/Dgd1.txt --method primal_dual
```

For more nodes, use sparse matrix:

```
python -m src.simplex.main_sparse tests/LMCF/PlanarNetworks/Cpl100.txt --format lmcf --demand-file tests/LMCF/PlanarNetworks/Dpl100.txt --time-limit 600
```
