# Exploiting Symmetry in Variational Quantum Machine Learning

A PennyLane + JAX implementation of symmetry-equivariant quantum classifiers, reproducing results from:

> Meyer et al., *Exploiting Symmetry in Variational Quantum Machine Learning*, PRX Quantum **4**, 010328 (2023). [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)

## Overview

This repository contains two notebooks implementing variational quantum circuits that exploit geometric symmetries to improve generalisation:

| Notebook | Symmetry | Task | Qubits |
|----------|----------|------|--------|
| `tic_tac.ipynb` | D₄ (dihedral) | 3-class classification | 9 |
| `Autonomous_Vehicle_Scenerios_Toy_Model.ipynb` | Z₄ (cyclic) | Regression | 9 |

Both use the "cemoid" gate topology (single-qubit RX + RY rotations followed by CRZ entangling gates) with data re-uploading, matching the paper's methodology.

---

## Tic-Tac-Toe Classifier

Classifies game outcomes (X wins, O wins, draw) on a 3×3 board. The D₄ symmetry — rotations and reflections of the square — partitions the 9 positions into three equivalence classes: corners, edges, and center.

| Model | Params/block | Architecture |
|-------|--------------|--------------|
| Equivariant | 9 | Shared weights across equivalent qubits |
| Non-equivariant | 34 | Independent params per qubit |

**Dataset:** 450 training / 300 validation samples (balanced classes)

---

## Autonomous Vehicle Scenarios

Predicts decision difficulty (0.0–1.0) for a simplified autonomous vehicle navigating road layouts. The model uses Z₄ symmetry (rotations only, no reflections) because mirroring a left-turn scenario produces a right-turn scenario with different difficulty.

| Model | Params/block | Total (l=3, p=1) |
|-------|--------------|------------------|
| Equivariant | 10 | 30 |
| Non-equivariant | 42 | 126 |

**Scenarios:** Straight roads, corners, T-crossings, and intersections with 6 difficulty levels

**Output:** ŷ = (Z_middle + 1) / 2 (center qubit expectation mapped to [0,1])

---

## Repository Structure

```
tic_tac.ipynb                              # D₄-equivariant tic-tac-toe classifier
Autonomous_Vehicle_Scenerios_Toy_Model.ipynb   # Z₄-equivariant vehicle scenario regressor
graphics/                                  # Diagrams (board indexing, road layouts)
```

## Getting Started

```bash
pip install pennylane jax jaxlib optax numpy matplotlib
```

Then open either notebook and run all cells. Training takes a few minutes on CPU.

## Key Concepts

**Data re-uploading:** Input features are encoded at the start of every layer, not just once. This increases expressivity while maintaining symmetry constraints.

**Equivariant circuits:** Qubits in the same equivalence class (e.g., all corners) share parameters, reducing the parameter count and enforcing that symmetric inputs produce symmetric outputs.

**Generalisation gap:** The difference between training and validation accuracy. Symmetric models show smaller gaps, indicating better generalisation from limited data.

## Reference

Meyer, J.J. et al., "Exploiting Symmetry in Variational Quantum Machine Learning",
PRX Quantum 4, 010328 (2023). [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)
