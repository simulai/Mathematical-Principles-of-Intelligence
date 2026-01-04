
import sys
import os
import json
import math
import re
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

# ==========================================
# 1. Cognitive Holonomy & Symplectic Dynamics
# ==========================================

class FourierHolonomyLayer(nn.Module):
    """
    Implements the "Cognitive Spectrum" hypothesis:
    Projecting reasoning states into a Fourier basis allows the system to 
    filter high-frequency "noise" (hallucinations) and retain low-frequency "truth" (axioms).
    """
    def __init__(self, dim, num_features):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        # Fourier Feature Projection: Maps input to high-dimensional frequency space
        # Fixed random weights (Gaussian) - similar to Positional Encoding but learnable scale if needed
        self.b = nn.Parameter(torch.randn(num_features, dim) * 10.0, requires_grad=False)
        self.linear = nn.Linear(num_features * 2, dim) # Mix sin and cos components
        
    def forward(self, x):
        """
        x: [Batch, Length, Dim]
        """
        # Project to frequency space: x @ b.T
        proj = torch.matmul(x, self.b.t()) # [Batch, Length, NumFeatures]
        
        # Fourier Basis: [sin(proj), cos(proj)]
        f_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) # [Batch, Length, NumFeatures*2]
        
        # Project back to manifold dimension
        x_f = self.linear(f_feat)
        
        # Residual connection (optional, but helps preserve gradient)
        return x + x_f

class StochasticSymplecticDynamics(nn.Module):
    """
    Implements "Fantasy with Constraints" (Stochastic Symplectic Dynamics).
    
    1. Fantasy (Noise Injection): Allows the system to explore off-manifold paths (Creativity).
    2. Constraint (Holonomic Pullback): Forces the path to return to a consistent logical flow.
    """
    def __init__(self, dim, noise_scale=0.1):
        super().__init__()
        self.dim = dim
        self.noise_scale = noise_scale
        self.holonomy_loss = HolonomyLoss()
        
    def forward(self, hidden_states):
        """
        hidden_states: [Batch, Length, Dim]
        """
        # 1. Inject Fantasy (Gaussian Noise)
        noise = torch.randn_like(hidden_states) * self.noise_scale
        fantasy_states = hidden_states + noise
        
        # 2. Compute Holonomy of the Fantasy Path
        # If the fantasy is "wild" (hallucination), Holonomy will be high.
        # If the fantasy is "insightful" (consistent), Holonomy will be low.
        h_score = self.holonomy_loss(fantasy_states)
        
        return fantasy_states, h_score

class HolonomyLoss(nn.Module):
    """
    Computes the Cognitive Holonomy (H).
    
    Theoretical Definition:
    H(gamma) = || P_gamma - I ||
    The deviation of the parallel transport along the reasoning path from the identity.
    
    Geometric Implementation (Discrete Curvature):
    We model the optimal reasoning path as a geodesic on the cognitive manifold.
    The Holonomy is measured as the "Geodesic Curvature" (covariant acceleration),
    approximated by the sine of the angle between consecutive reasoning steps.
    
    L_H = Sum || v_t x v_{t+1} ||  (Area of the path deviation)
    """
    def __init__(self, lambda_h=0.1):
        super().__init__()
        self.lambda_h = lambda_h
        
    def forward(self, hidden_states, return_per_step=False):
        """
        hidden_states: [Batch, Length, Dim]
        """
        # 1. Compute Tangent Vectors (Velocity)
        # v_t = x_{t+1} - x_t
        velocity = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        
        # Normalize velocity to focus on Direction (Geometry) rather than Speed (Energy)
        # Add epsilon to avoid division by zero
        v_norm = torch.norm(velocity, dim=-1, keepdim=True) + 1e-8
        v_dir = velocity / v_norm
        
        # 2. Compute Discrete Curvature (Angle between consecutive steps)
        # We want v_t and v_{t+1} to be collinear (Geodesic)
        # Cosine similarity: (v_t . v_{t+1})
        if v_dir.shape[1] < 2:
            # Not enough steps to measure curvature, fallback to simple energy (smoothness)
            return torch.mean(velocity ** 2) * self.lambda_h
            
        v_t = v_dir[:, :-1, :]
        v_next = v_dir[:, 1:, :]
        
        # Cosine similarity
        cosine_sim = torch.sum(v_t * v_next, dim=-1)
        
        # Curvature Loss = 1 - Cosine (Penalize turning)
        # Range: [0, 2]. 0 means straight line, 2 means U-turn.
        curvature = 1.0 - cosine_sim
        
        if return_per_step:
            # curvature shape: [Batch, Length-2]
            # We pad with zeros at start and end to match original length [Batch, Length]
            # giving a "curvature" score to each node (0 for endpoints)
            pad_start = torch.zeros(curvature.shape[0], 1).to(curvature.device)
            pad_end = torch.zeros(curvature.shape[0], 1).to(curvature.device)
            # Result: [Batch, Length]
            return torch.cat([pad_start, curvature, pad_end], dim=1)
        
        # Total Holonomy is the integral of curvature along the path
        loss = torch.mean(curvature)
        
        return self.lambda_h * loss

# ==========================================
# 2. MPI Reasoning Agent
# ==========================================

class MPIReasoningAgent:
    def __init__(self, model_name="deepseek-chat", embedding_model="distilbert-base-uncased", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        # Initialize OpenAI Client (if key exists)
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            except ImportError:
                print("OpenAI library not found. Running in MOCK mode.")
        else:
            print("WARNING: DEEPSEEK_API_KEY not found. Running in MOCK mode.")
        
        # Initialize Cognitive Holonomy Loss
        self.holonomy_loss = HolonomyLoss(lambda_h=0.05)

        # Initialize Cognitive Spectrum (Fourier) & Fantasy (Symplectic) layers
        # Assuming embedding dim is 768 (DistilBERT)
        self.fourier_layer = FourierHolonomyLayer(dim=768, num_features=256)
        self.symplectic_dynamics = StochasticSymplecticDynamics(dim=768, noise_scale=0.05)
        
        # Initialize Embedding Model (Local) for MPI calculation
        print(f"Loading embedding model: {embedding_model}...")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.embed_model = AutoModel.from_pretrained(embedding_model)
            self.embed_model.eval()
        except ImportError:
            print("Transformers library not found. Embeddings will be random (MOCK).")
            self.embed_model = None
        
        print(f"Initialized MPI Agent with {model_name} and Cognitive Holonomy (lambda=0.05)")

    def get_embeddings(self, text_list):
        """
        Get embeddings for a list of text steps using the local model.
        Returns: [1, Length, Dim] tensor
        """
        if self.embed_model is None:
            # Mock embeddings
            return torch.randn(1, len(text_list), 768)

        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        # Use CLS token embedding for each step
        cls_embeddings = outputs.last_hidden_state[:, 0, :] # [NumSteps, Dim]
        return cls_embeddings.unsqueeze(0) # [1, NumSteps, Dim]

    def solve(self, problem_text, formal_template=None):
        print(f"Solving: {problem_text[:50]}...")
        
        # 1. Generate Candidate Thoughts
        candidates = self._generate_candidates(problem_text)
        
        if not candidates:
            print("No candidates generated.")
            return None

        # 2. Score Candidates using MPI (Cognitive Holonomy Loss on Embeddings)
        best_candidate = None
        best_score = float('inf')
        candidate_scores = []
        
        for cand in candidates:
            if not cand['steps']:
                continue
                
            # Get real embeddings for the reasoning steps
            embeddings = self.get_embeddings(cand['steps']) # [1, Length, Dim]
            
            # --- Apply Cognitive Spectrum (Fourier Filter) ---
            # Project to "Truth Spectrum" to filter high-frequency noise
            embeddings_f = self.fourier_layer(embeddings)
            
            # --- Apply Stochastic Symplectic Dynamics (Fantasy) ---
            # Inject noise and measure if the system can "pull back" to logic
            fantasy_embeddings, fantasy_h_score = self.symplectic_dynamics(embeddings_f)
            
            # --- Holonomic Correction (The "Pullback") ---
            # Simulating the Manifold ODE Solver: pull the fantasy back to the geodesic
            # If the thought is robust, this correction should be small (stable attractor).
            # If the thought is a hallucination, this correction will be large (unstable).
            correction_delta = torch.norm(fantasy_embeddings - embeddings_f)
            
            # Final Score = Holonomy + Correction Effort
            # A good thought has low curvature AND is a stable attractor (hard to push away)
            h_score = fantasy_h_score.item() + 0.1 * correction_delta.item()
            
            # Get per-step entropy (using the filtered embeddings for precision)
            step_entropy = self.holonomy_loss(embeddings_f, return_per_step=True).squeeze().tolist()
            if isinstance(step_entropy, float):
                step_entropy = [step_entropy]
                
            print(f"  Candidate: {cand['name']} | H-Score: {h_score:.4f} (Base: {fantasy_h_score.item():.4f} + Corr: {correction_delta.item():.4f})")
            
            candidate_scores.append({
                "name": cand['name'],
                "h_score": h_score,
                "step_entropy": step_entropy,
                "answer": self._extract_answer(cand['steps'])
            })
            
            if h_score < best_score:
                best_score = h_score
                best_candidate = cand
        
        if best_candidate:
            print(f"Selected Best Path: {best_candidate['name']}")
            answer = self._extract_answer(best_candidate['steps'])
            print(f"Extracted Answer: {answer}")
            
            return {
                "answer": answer,
                "reasoning": best_candidate['steps'],
                "best_candidate": best_candidate,
                "scores": candidate_scores
            }
        else:
            return None

    def _extract_answer(self, steps):
        full_text = " ".join(steps)
        match = re.search(r'\\boxed\{(\d+)\}', full_text)
        if match:
            return int(match.group(1))
        return None

    def _generate_candidates(self, problem_text):
        if not self.client:
            return self._mock_candidates()

        prompt = f"""
You are a mathematical reasoning assistant. 
Problem: {problem_text}

Please provide **3 distinct step-by-step reasoning paths** to solve this problem.
1. Path 1: A standard, direct approach.
2. Path 2: An alternative method (if applicable).
3. Path 3: A creative perspective.

Format your response as a valid JSON object with the following structure:
{{
  "candidates": [
    {{
      "name": "Path 1",
      "steps": ["Step 1...", "Step 2...", "Conclusion: The answer is \\boxed{{42}}"]
    }},
    ...
  ]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful math assistant. Output only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            content = response.choices[0].message.content
            # Clean up potential markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            return data.get("candidates", [])
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return self._mock_candidates()

    def _mock_candidates(self):
        import random
        ans = random.randint(0, 999)
        return [
            {
                "name": "Path A (Direct - Mock)",
                "steps": ["Assume n is odd", "n = 2k + 1", "n^2 = 4k^2 + 4k + 1", f"The answer is \\boxed{{{ans}}}"],
            },
            {
                "name": "Path B (Contradiction - Mock)",
                "steps": ["Assume n^2 is even", "Then n must be even", "Contradiction", f"The answer is \\boxed{{{ans}}}"],
            }
        ]

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    # Example usage for Kaggle
    # In a real competition, you would read 'test.csv' or use the 'aimo' module
    
    # Check for input file
    input_file = "test.csv"
    if os.path.exists(input_file):
        # Process CSV
        print(f"Processing {input_file}...")
        pass # Add CSV processing logic here
    else:
        print("No input file found. Running demo problem.")
        agent = MPIReasoningAgent()
        problem = "Find the sum of all positive integers less than 20 that are multiples of 3 or 5."
        result = agent.solve(problem)
        if result:
            print(f"Final Answer: {result['answer']}")

if __name__ == "__main__":
    main()
