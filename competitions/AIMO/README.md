# AI Mathematical Olympiad (AIMO) - MPI Strategy

This directory contains the experimental framework for applying the **Mathematical Principles of Intelligence (MPI)** to the [AI Mathematical Olympiad](https://kaggle.com/competitions/ai-mathematical-olympiad-prize).

## 🏆 Goal
Solve high-difficulty math problems (AIME/IMO level) by combining **LLM Reasoning** with **Formal Verification (Lean 4)**, guided by **MPI Thermodynamics**.

## 🧠 The MPI Advantage

Standard LLMs suffer from "hallucination" (high entropy generation). We use MPI principles to constrain the reasoning process:

1.  **SPHA (e-base Attention)**: Optimal information branching. We limit the "Thought Tree" branching factor to $b \approx e$ (2 or 3 distinct reasoning paths) to maximize efficiency.
2.  **Cognitive Holonomy ($\mathcal{H}$)**: We treat mathematical proof as a "flow" on a cognitive manifold.
    *   **Low $\mathcal{H}$ Violation**: Smooth, logical deduction (preferred).
    *   **High $\mathcal{H}$ Violation**: Logical leaps or non-sequiturs (pruned).
3.  **Formal Verification**: The "Zero-Dissipation Limit". A verified Lean 4 proof has zero entropy (perfect certainty).

## 🛠️ Architecture

```mermaid
graph TD
    A[Problem Input] --> B{MPI-Reasoning Agent}
    B -->|Generate| C[Thought Tree (b=3)]
    C -->|Score| D[Cognitive Holonomy Filter]
    D -->|Prune| E[Best Reasoning Path]
    E -->|Formalize| F[Lean 4 Code Generator]
    F -->|Verify| G{Lean Compiler}
    G -->|Success| H[Output Answer]
    G -->|Fail| I[Feedback Loop -> B]
```

## 📂 Structure

*   `dataset/`: Sample AIME/IMO problems.
*   `lean_solver/`: Lean 4 project for formalizing solutions.
*   `agent/`: Python code for the Reasoning Agent (LLM + MPI metrics).
    *   `reasoning_agent.py`: Main agent logic (DeepSeek + MPI).
*   `evaluate.py`: Main script to run the pipeline.

## 🚀 Getting Started

### Prerequisites
1.  **Python 3.10+**
2.  **Lean 4**: Ensure `elan` and `lake` are in your PATH.
3.  **DeepSeek API Key**: Set `DEEPSEEK_API_KEY` in `.env`.

### Installation
```bash
pip install -r requirements.txt
```

### Usage
Run the Reasoning Agent to solve sample problems:
```bash
cd agent
python reasoning_agent.py
```

## 📝 Roadmap

- [x] **Phase 1**: Setup Lean 4 environment and "Hello World" proof.
- [x] **Phase 2**: Implement "Autoformalization" (Natural Language -> Lean 4) using DeepSeek.
- [x] **Phase 3**: Integrate MPI metrics (Zhang Invariant) to select best reasoning paths.
- [ ] **Phase 4**: Full loop verification (Automated Lean Compilation).
- [ ] **Phase 5**: Scale up to AIME 2024 problems.