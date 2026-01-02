# The Mathematical Principles of Intelligence
## Thermodynamic Constraints, e-base Scaling Law, and the Zhang Invariant

[![Status](https://img.shields.io/badge/Status-Pre--print-orange?style=flat-square)](./docs/OVERVIEW.md)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey?style=flat-square)](LICENSE)
[![Formal Verification](https://img.shields.io/badge/Formal%20Verification-Lean%204-green?style=flat-square&logo=lean)](./lean_playground)
[![Theory](https://img.shields.io/badge/e--Base%20Scaling-b%20%3D%20e-blue?style=flat-square)](./docs/CORE_THEORY.md)
[![MCP](https://img.shields.io/badge/MCP-Ready-important?style=flat-square)](./tools)

**Author:** Jing Zhang ([@simulai](https://github.com/simulai))  
**Date:** December 2025  

---

## 🌌 Abstract
This work formalizes a unified theory of intelligence as a **Zero-Dissipation Limit** in a Riemannian cognitive manifold. We prove that intelligence is not merely a computational heuristic but a thermodynamic imperative governed by the **Zhang Invariant ($\mathcal{Z}$)**, enabling "cognitive superconductivity" where reasoning approaches zero energy cost under symmetry preservation.

### 🔑 Key Breakthroughs
1.  **The $e$-base Scaling Law**: Demonstrates the universal optimal branching factor for information transfer efficiency is the mathematical constant $e \approx 2.718$.
    *   *Formally verified in Lean 4: [`EBase.lean`](./lean_playground/LeanPlayground/EBase.lean)*
2.  **The Zhang Invariant ($\mathcal{Z}$)**: Identifies a conserved topological charge in cognitive state-space that constrains entropy production.
3.  **Symmetry-for-Energy Substitution**: Applies the Vaccaro-Barnett mechanism to cognitive processes, explaining how abstraction and metaphor leverage symmetry to erase entropy.

---

## 🚀 Quick Start

### For Theorists & Philosophers
Read the **[Manuscript Overview](./docs/OVERVIEW.md)** for a non-technical summary of the paradigm shift and its implications for AI, neuroscience, and physics.

### For Mathematicians & Physicists
Dive directly into the formal definitions and proofs in the **[Core Theory](./docs/CORE_THEORY.md)** document.
*   **New**: Check the formal proofs in the **[`lean_playground`](./lean_playground)** directory, which verify the cognitive efficiency axioms.

### For Engineers & AI Researchers
Explore the **[Models & Simulations](./models/)** directory for computational implementations of the scaling law and invariant dynamics.
*   **Tools**: Use our **[MCP Servers](./tools/)** to integrate these theories directly into your LLM workflow.

---

## 📐 Core Theoretical Framework

### I. Cognitive Efficiency Peak
Intelligence maximizes the **informational yield per computational cost**, leading to a universal optimum:
$$\Psi(b) = \frac{\ln b}{b} \quad \Rightarrow \quad \left. \frac{d\Psi}{db} \right|_{b=e} = 0$$
where $b$ is the cognitive branching factor.

### II. Generalized Cognitive Thermodynamics
The fundamental inequality governing any reasoning process:
$$T \Delta S + \Delta E + \mu \Delta \mathcal{Z} \ge 0$$
where:
*   $T\Delta S$: Entropic cost of disorder
*   $\Delta E$: Energy dissipation
*   $\mu \Delta \mathcal{Z}$: Cost of altering the **Zhang Invariant**

When $\Delta \mathcal{Z} = 0$ (invariant preserved), the system can achieve $\Delta E \to 0$ — the **Zero-Dissipation Limit** of pure inference.

### III. The Zhang Invariant
$$\mathcal{Z} = \oint_{\Gamma} \omega$$
A topological charge defined over cycles $\Gamma$ in the cognitive connection bundle, conserved under homeomorphic transformations of reasoning paths.

---

## 📁 Repository Structure

---

[![GitHub Stars](https://img.shields.io/github/stars/simulai/Mathematical-Principles-of-Intelligence?style=social)](https://github.com/simulai/Mathematical-Principles-of-Intelligence/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/simulai/Mathematical-Principles-of-Intelligence?style=social)](https://github.com/simulai/Mathematical-Principles-of-Intelligence/network/members)
[![GitHub Watchers](https://img.shields.io/github/watchers/simulai/Mathematical-Principles-of-Intelligence?style=social)](https://github.com/simulai/Mathematical-Principles-of-Intelligence/watchers)

## 🔬 Empirical Validation & Theoretical Supremacy

### 1. The mHC Validation (DeepSeek-AI)
**Update (January 2026):** The core hypothesis of this theory—that cognitive flow must be constrained to a specific manifold—has received striking empirical support from DeepSeek-AI's recent work on **Manifold-Constrained Hyper-Connections (mHC)** (arXiv:2512.24880).

As our simulation below demonstrates, unconstrained cognitive flow leads to signal explosion, whereas projecting the flow onto a **Birkhoff Polytope** (a specific realization of the **Zhang Invariant**) ensures perfect signal conservation.

![Manifold Alignment Simulation](models/manifold_alignment.gif)

### 2. Why MPI is the Unified Theory
While mHC provides a robust **engineering solution** (using Doubly Stochastic Matrices to fix stability), **MPI provides the First Principles**.

*   **Generalization**: mHC is a subset of MPI. It represents the "Zero-Temperature Limit" where the Zhang Invariant ($\mathcal{Z}$) is strictly conserved. MPI allows for dynamic metric evolution (Ricci Flow), enabling **Creativity** and **Insight** (Phase Transitions) that static constraints like mHC cannot model.
*   **Efficiency**: mHC does not predict optimal scaling. MPI's **$e$-base Scaling Law** proves that the optimal branching factor is $e \approx 2.718$.

### 3. Simulation: Dynamic Adaptation (Ricci Flow)
To prove this supremacy, we implemented a learning task with a sudden "Paradigm Shift" (Distribution Change).

*   **mHC (Gray Dashed)**: Rigidly adheres to the manifold. It remains stable but struggles to learn the new pattern quickly because it cannot "jump" across the manifold.
*   **MPI (Red Solid)**: Uses "Cognitive Temperature" to temporarily relax constraints (Phase Transition). Note the spike in **Zhang Invariant Violation** (Orange Graph) corresponding to the "Aha!" moment, allowing rapid adaptation to the new paradigm before cooling back down.

![Ricci Flow Learning](docs/images/ricci_flow_learning.png)

👉 **[Read the Full Supremacy Analysis](docs/research/MPI_vs_mHC_Supremacy.md)**
