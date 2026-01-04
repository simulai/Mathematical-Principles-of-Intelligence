import sys
import os
import json
import math
import re
import torch
import torch.nn as nn
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
try:
    from lean_verifier import LeanVerifier
except ImportError:
    # Fallback for relative import if run from different location
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from lean_verifier import LeanVerifier

# Load environment variables
load_dotenv()

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

class MPIReasoningAgent:
    def __init__(self, model_name="deepseek-chat", embedding_model="distilbert-base-uncased"):
        self.model_name = model_name
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            print("WARNING: DEEPSEEK_API_KEY not found in environment variables.")
            print("Please set it in .env or environment. Running in MOCK mode.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        # Initialize Cognitive Holonomy Loss
        self.holonomy_loss = HolonomyLoss(lambda_h=0.05)
        
        # Initialize Lean Verifier
        # Assuming the lean_solver directory is at ../lean_solver relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lean_project_path = os.path.join(base_dir, "lean_solver")
        self.verifier = LeanVerifier(lean_project_path)
        
        # Initialize Embedding Model (Local) for MPI calculation
        print(f"Loading embedding model: {embedding_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model)
        self.embed_model.eval()
        
        print(f"Initialized MPI Agent with {model_name} and Cognitive Holonomy (lambda=0.05)")

    def load_problems(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_embeddings(self, text_list):
        """
        Get embeddings for a list of text steps using the local model.
        Returns: [1, Length, Dim] tensor
        """
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        # Use CLS token embedding for each step
        # outputs.last_hidden_state is [Batch, SeqLen, Dim]
        # We want one vector per step.
        # Since text_list is a list of steps, inputs['input_ids'] has shape [NumSteps, SeqLen]
        # We take the CLS token (index 0) from each step
        cls_embeddings = outputs.last_hidden_state[:, 0, :] # [NumSteps, Dim]
        return cls_embeddings.unsqueeze(0) # [1, NumSteps, Dim]

    def solve(self, problem):
        print(f"Solving: {problem['problem']}")
        
        # 1. Generate Candidate Thoughts
        candidates = self._generate_candidates(problem['problem'])
        
        if not candidates:
            print("No candidates generated.")
            return None

        # 2. Score Candidates using MPI (Cognitive Holonomy Loss on Embeddings)
        best_candidate = None
        best_score = float('inf')
        
        # Store all scores for "Phase Transition" analysis
        candidate_scores = []
        
        for cand in candidates:
            if not cand['steps']:
                continue
                
            # Get real embeddings for the reasoning steps
            embeddings = self.get_embeddings(cand['steps'])
            
            # Calculate Cognitive Holonomy Loss (lower is better, means more coherent flow)
            h_score = self.holonomy_loss(embeddings).item()
            
            # Get per-step entropy production for visualization
            step_entropy = self.holonomy_loss(embeddings, return_per_step=True).squeeze().tolist()
            if isinstance(step_entropy, float):
                step_entropy = [step_entropy] # Handle single step case
                
            print(f"  Candidate: {cand['name']} | H-Score: {h_score:.4f}")
            
            candidate_scores.append({
                "name": cand['name'],
                "h_score": h_score,
                "step_entropy": step_entropy, # Store trajectory
                "answer": self._extract_answer(cand['steps'])
            })
            
            if h_score < best_score:
                best_score = h_score
                best_candidate = cand
        
        if best_candidate:
            print(f"Selected Best Path: {best_candidate['name']}")
            
            # Extract Answer
            answer = self._extract_answer(best_candidate['steps'])
            print(f"Extracted Answer: {answer}")

            # 3. Formalize using LLM (if template exists or we can generate one)
            lean_code = None
            verification_success = False
            
            if 'formal_statement_template' in problem and problem['formal_statement_template']:
                lean_code = self._formalize_with_llm(best_candidate, problem)
                
                # 4. Verify Code
                print("Verifying Lean code...")
                success, message = self.verifier.verify(lean_code)
                verification_success = success
                
                if success:
                    print("Verification SUCCESS!")
                else:
                    print(f"Verification FAILED. Error:\n{message}")
                    # Attempt to fix
                    print("Attempting to fix code...")
                    lean_code = self._fix_code_with_llm(lean_code, message, problem)
                    success_fix, message_fix = self.verifier.verify(lean_code)
                    
                    if success_fix:
                        print("Fix SUCCESS!")
                        verification_success = True
                    else:
                        print(f"Fix FAILED. Error:\n{message_fix}")
            else:
                print("No formal statement template provided. Skipping formal verification.")
            
            return {
                "answer": answer,
                "lean_code": lean_code,
                "verified": verification_success,
                "reasoning": best_candidate['steps'],
                "all_scores": candidate_scores # Return all scores for analysis
            }
        else:
            print("Failed to select a best candidate.")
            return None

    def _extract_answer(self, steps):
        # Look for \boxed{number} in the last few steps
        full_text = " ".join(steps)
        match = re.search(r'\\boxed\{(\d+)\}', full_text)
        if match:
            return int(match.group(1))
        # Fallback: look for just a number at the end
        # This is risky, but better than nothing
        return None

    def _fix_code_with_llm(self, code, error_message, problem):
        if not self.api_key:
            return code # Cannot fix without LLM

        print("Calling DeepSeek to fix Lean code...")
        
        prompt = f"""
You are an expert Lean 4 developer.
The following Lean 4 code failed to compile.

Problem:
{problem['problem']}

Code:
```lean
{code}
```

Error Message:
{error_message}

Please fix the code. Output ONLY the fixed Lean 4 code. No markdown explanations.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert Lean 4 developer. Output only raw code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            fixed_code = response.choices[0].message.content
            # Clean up potential markdown code blocks
            if "```lean" in fixed_code:
                fixed_code = fixed_code.split("```lean")[1].split("```")[0]
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0]
            return fixed_code.strip()
        except Exception as e:
            print(f"Error calling DeepSeek API (Fixing): {e}")
            return code

    def _generate_candidates(self, problem_text):
        if not self.api_key:
            # Fallback to mock if no API key
            print("Using MOCK generation (No API Key)")
            return self._mock_candidates()

        prompt = f"""
You are a mathematical reasoning assistant. 
Problem: {problem_text}

Please provide **3 distinct step-by-step reasoning paths** to solve this problem.
1. Path 1: A standard, direct approach.
2. Path 2: An alternative method (if applicable) or a verification path.
3. Path 3: A creative or different perspective (but still logical).

The paths should be logically sound and break down the proof into small, verifiable steps.
Crucial: The final answer must be a non-negative integer.
At the end of your reasoning, strictly output the final answer in LaTeX boxed format, e.g., \\boxed{{42}}.

Format your response as a valid JSON object with the following structure:
{{
  "candidates": [
    {{
      "name": "Path 1 (Strategy Name)",
      "steps": ["Step 1...", "Step 2...", "Conclusion: The answer is \\boxed{{42}}"]
    }},
    {{
      "name": "Path 2 (Strategy Name)",
      "steps": ["Step 1...", "Step 2...", "Conclusion: The answer is \\boxed{{42}}"]
    }},
    {{
      "name": "Path 3 (Strategy Name)",
      "steps": ["Step 1...", "Step 2...", "Conclusion: The answer is \\boxed{{42}}"]
    }}
  ]
}}
Do not include any markdown formatting or extra text outside the JSON.
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
            # Clean up potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            return data.get("candidates", [])
        except Exception as e:
            print(f"Error calling DeepSeek API (Generation): {e}")
            # Debug: print content snippet
            try:
                print(f"Raw content snippet: {content[:200]}...")
            except:
                pass
            return self._mock_candidates()

    def _formalize_with_llm(self, candidate, problem):
        if not self.api_key:
            return self._formalize_template(candidate, problem['formal_statement_template'])

        print(f"Formalizing strategy: {candidate['name']}...")
        
        prompt = f"""
You are an expert Lean 4 theorem prover.
Your task is to translate a natural language proof into a formal Lean 4 proof.

Problem Statement:
{problem['problem']}

Formal Statement Template:
{problem['formal_statement_template']}

Selected Reasoning Strategy:
{json.dumps(candidate['steps'], indent=2)}

Instructions:
1. Use the provided Formal Statement Template.
2. Replace the 'sorry' with a valid Lean 4 proof based on the Reasoning Strategy.
3. Use 'import Mathlib' at the top.
4. Output ONLY the Lean 4 code. No markdown, no explanations.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert Lean 4 developer. Output only raw code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Lower temperature for code generation
                max_tokens=2000
            )
            code = response.choices[0].message.content
            # Clean up potential markdown code blocks
            if "```lean" in code:
                code = code.split("```lean")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            print(f"Error calling DeepSeek API (Formalization): {e}")
            return self._formalize_template(candidate, problem['formal_statement_template'])

    def _mock_candidates(self):
        import random
        ans = random.randint(0, 999)
        return [
            {
                "name": "Path A (Direct - Mock)",
                "steps": ["Assume n is odd", "n = 2k + 1", "n^2 = 4k^2 + 4k + 1", "n^2 = 2(2k^2 + 2k) + 1", f"The answer is \\boxed{{{ans}}}"],
            },
            {
                "name": "Path B (Contradiction - Mock)",
                "steps": ["Assume n^2 is even", "Then n must be even", "Contradiction", f"The answer is \\boxed{{{ans}}}"],
            },
            {
                "name": "Path C (Confused - Mock)",
                "steps": ["n is odd", "maybe n is prime?", "primes are odd usually", "so n^2 is prime?", f"I guess \\boxed{{{ans}}}"],
            }
        ]

    def _formalize_template(self, candidate, template):
        # Simple string injection
        proof_body = "\n  ".join([f"-- {step}" for step in candidate['steps']])
        code = f"""
import Mathlib

{template}
  -- Proof generated by MPI Agent
  -- Strategy: {candidate['name']}
  {proof_body}
  sorry
"""
        return code

if __name__ == "__main__":
    agent = MPIReasoningAgent()
    
    # Ensure dataset exists
    dataset_path = "../dataset/sample_problems.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
    else:
        problems = agent.load_problems(dataset_path)
        for p in problems:
            solution = agent.solve(p)
            if solution:
                print("\nGenerated Lean Code:")
                print(solution)
                print("-" * 40)
