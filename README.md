# Exploiting Symmetry in Variational Quantum Machine Learning

A PennyLane + JAX implementation of symmetry-equivariant quantum circuits, reproducing results from:

> Meyer et al., *Exploiting Symmetry in Variational Quantum Machine Learning*, PRX Quantum **4**, 010328 (2022). [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)

## Overview

This repository contains three notebooks implementing variational quantum circuits that exploit symmetries to improve generalisation and trainability:

| Notebook | Symmetry | Task | Qubits |
|----------|----------|------|--------|
| `tic_tac.ipynb` | D₄ (dihedral) | 3-class classification | 9 |
| `Autonomous_Vehicle_Scenerios_Toy_Model.ipynb` | Z₄ (cyclic) | Regression | 9 |
| `transverse-field-ising-model.ipynb` | Z₂ (parity) | Ground State Search (VQE) | Variable (e.g. 10) |

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

## Transverse-Field Ising Model (TFIM)

Solves for the ground state energy of a 1D spin chain using the Variational Quantum Eigensolver (VQE). The Hamiltonian possesses a global $\mathbb{Z}_2$ parity symmetry ($P = \prod X_i$).

| Model | Gates Used | Characteristics |
|-------|------------|-----------------|
| Equivariant (QAOA) | RX, ZZ | Preserves parity; avoids barren plateaus at high depth |
| Non-equivariant (QAOA') | RX, RY, ZZ | Breaks parity via RY; suffers from "false convergence" |

**Experiment:** Compares convergence rates and energy error as circuit depth ($p$) increases.

**Key Finding:** For deep circuits ($p \ge N/2$), the equivariant model reliably reaches the ground state, whereas the non-equivariant model gets stuck in local minima (barren plateaus) due to searching unphysical symmetry sectors.

---

## Repository Structure

```
tic_tac.ipynb                                  # D₄-equivariant tic-tac-toe classifier
Autonomous_Vehicle_Scenerios_Toy_Model.ipynb   # Z₄-equivariant vehicle scenario regressor
transverse-field-ising-model.ipynb             # Z₂-equivariant VQE for TFIM graphics/
graphics                                       # Diagrams (board indexing, road layouts)
```

## Getting Started

```bash
pip install pennylane jax jaxlib optax numpy matplotlib
```

Then open either notebook and run all cells. Training takes a few minutes on CPU.

## Key Concepts

**Equivariant Circuits:** Constraining the quantum model to respect the symmetry of the problem.
* **For QML (Geometric Symmetries):** Qubits in the same spatial equivalence class (e.g., all corners of a board) share parameters. This reduces the parameter count and ensures symmetric inputs produce symmetric outputs.
* **For VQE (Internal Symmetries):** The ansatz is constructed using only gates that commute with the symmetry operator (e.g., $P = \prod X$). This restricts the search to the physical Hilbert space sector (e.g., positive parity), preventing the ansatz from exploring unphysical states.

**Inductive Bias & Trainability:**
* **In QML:** Symmetry provides a geometric inductive bias, leading to a smaller **generalisation gap** (the difference between training and validation accuracy), allowing models to learn from less data.
* **In VQE:** Symmetry acts as a physical inductive bias. It transforms the optimization landscape from a flat **Barren Plateau** (characteristic of non-equivariant models at high depth) into a navigable gorge, allowing the optimizer to converge to the true ground state.

**Data Re-uploading (QML only):** In the classification/regression tasks, input features are encoded at the start of every layer, not just once. This increases expressivity without breaking the symmetry constraints defined by the gate topology.
## Reference

Meyer, J.J. et al., "Exploiting Symmetry in Variational Quantum Machine Learning",
PRX Quantum 4, 010328 (2022). [arXiv:2205.06217](https://arxiv.org/abs/2205.06217)

