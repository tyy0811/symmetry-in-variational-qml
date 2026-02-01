# Exploiting Symmetry in Variational Quantum Machine Learning

A PennyLane + JAX implementation of the D4-equivariant quantum classifier for tic-tac-toe, reproducing results from:

> Meyer et al., *Exploiting Symmetry in Variational Quantum Machine Learning*, PRX Quantum **4**, 010328 (2023). [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)

## Overview

The notebook implements a variational quantum circuit that classifies tic-tac-toe game outcomes (X wins, O wins, draw) using 9 qubits. The key idea is that the tic-tac-toe board has D4 (dihedral) symmetry — rotations and reflections of the square — which partitions the 9 board positions into three equivalence classes: corners, edges, and center. By constraining the circuit to respect this symmetry, the model generalises better with fewer parameters.

| Model | Params | Architecture |
|-------|--------|-------------|
| Equivariant | 18 | 9 shared params/layer × 2 layers |
| Non-equivariant | 68 | 34 independent params/layer × 2 layers |

Both models use the same "cemoid" gate topology (single-qubit RX + RY rotations followed by CRY entangling gates) with data re-uploading at every layer, matching the paper's Figure 3.

## Repository structure

```
tic_tac.ipynb       # Main notebook
graphics/           # Diagrams (board indexing scheme)
```

## Getting started

```bash
pip install pennylane jax jax.numpy optax matplotlib numpy random time
```

Then open `tic_tac.ipynb` and run all cells. Training takes a few minutes on CPU.

## Notebook outline

1. **Game engine** — random tic-tac-toe play and balanced dataset generation (450 training / 300 validation samples)
2. **D4 symmetry analysis** — group generators, equivalence classes, twirling formula
3. **Equivariant circuit** — 9 params/layer with shared weights across equivalent qubits
4. **Non-equivariant circuit** — 34 independent params/layer, same gate topology
5. **Training** — Adam optimiser, MSE loss, 100 epochs × 30 mini-batches of 15
6. **Comparison** — validation accuracy and generalisation gap between models

## Reference

Meyer, J.J. et al., "Exploiting Symmetry in Variational Quantum Machine Learning",
PRX Quantum 4, 010328 (2023). arXiv:2205.06217

