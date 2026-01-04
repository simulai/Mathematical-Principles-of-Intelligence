import torch
import torch.nn as nn
import unittest
import numpy as np
from quantum_echt_prototype import DensityMatrixEmbedding, LindbladEvolution, QuantumMeasurement

class TestQuantumECHT(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.embed_dim = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Testing on device: {self.device}")

    def test_density_matrix_embedding(self):
        print("\n=== Testing DensityMatrixEmbedding ===")
        embedding = DensityMatrixEmbedding(self.embed_dim).to(self.device)
        x = torch.randn(self.batch_size, self.embed_dim).to(self.device)
        
        rho = embedding(x)
        
        # Check shape
        self.assertEqual(rho.shape, (self.batch_size, self.embed_dim, self.embed_dim))
        
        # Check trace = 1
        # trace of each matrix in batch
        traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
        print(f"Traces: {traces}")
        self.assertTrue(torch.allclose(traces, torch.ones_like(traces), atol=1e-5))
        
        # Check Hermitian (approx for real/complex)
        # rho should be Hermitian: rho == rho.conj().t()
        # For batched: rho[b] == rho[b].conj().t()
        for i in range(self.batch_size):
            r = rho[i]
            self.assertTrue(torch.allclose(r, r.T.conj(), atol=1e-5))
            
        # Check Positive Semi-Definite
        # Eigenvalues should be >= 0
        # Since rho might be real here (before evolution), we can just check eigvals.
        # Note: torch.linalg.eigvalsh is for Hermitian matrices
        eigvals = torch.linalg.eigvalsh(rho)
        print(f"Min Eigenvalue: {eigvals.min().item()}")
        self.assertTrue((eigvals >= -1e-5).all())

    def test_lindblad_evolution(self):
        print("\n=== Testing LindbladEvolution ===")
        # Setup
        embedding = DensityMatrixEmbedding(self.embed_dim).to(self.device)
        evolution = LindbladEvolution(self.embed_dim, dt=0.01).to(self.device)
        
        x = torch.randn(self.batch_size, self.embed_dim).to(self.device)
        rho_init = embedding(x)
        
        # Evolve
        # Since rho_init is real, and evolution introduces imaginary parts, output should be complex
        rho_next = evolution(rho_init)
        
        print(f"Rho init type: {rho_init.dtype}")
        print(f"Rho next type: {rho_next.dtype}")
        
        self.assertTrue(rho_next.is_complex())
        
        # Check Trace Preservation (Lindblad should preserve trace)
        # d_rho = -i[H, rho] + L(rho)
        # tr(d_rho) = -i tr(H rho - rho H) + tr(L rho L^d - 0.5 {L^d L, rho})
        # tr([H, rho]) = 0
        # tr(L rho L^d) = tr(L^d L rho)
        # tr(0.5 {L^d L, rho}) = 0.5 (tr(L^d L rho) + tr(rho L^d L)) = tr(L^d L rho)
        # So trace should be preserved exactly in continuous time.
        # In discrete time (Euler step), it might drift slightly O(dt^2).
        
        traces_init = torch.diagonal(rho_init, dim1=-2, dim2=-1).sum(-1)
        traces_next = torch.diagonal(rho_next, dim1=-2, dim2=-1).sum(-1)
        
        print(f"Traces init: {traces_init}")
        print(f"Traces next (real part): {traces_next.real}")
        
        # Euler integration is not strictly trace preserving for finite dt, but should be close
        self.assertTrue(torch.allclose(traces_next.real, torch.ones_like(traces_init), atol=1e-2))
        
        # Check Hermitian
        for i in range(self.batch_size):
            r = rho_next[i]
            # r.T.conj()
            self.assertTrue(torch.allclose(r, r.T.conj(), atol=1e-5))

    def test_quantum_measurement(self):
        print("\n=== Testing QuantumMeasurement ===")
        # Setup
        embedding = DensityMatrixEmbedding(self.embed_dim).to(self.device)
        evolution = LindbladEvolution(self.embed_dim).to(self.device)
        measurement = QuantumMeasurement(self.embed_dim).to(self.device)
        
        x = torch.randn(self.batch_size, self.embed_dim).to(self.device)
        rho = evolution(embedding(x))
        
        output = measurement(rho)
        
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output}")
        
        self.assertEqual(output.shape, (self.batch_size,))
        self.assertFalse(output.is_complex())
        
        # Probabilities should be roughly in [0, 1] if measurement vectors are normalized?
        # My implementation: P = <u|rho|u>
        # u is normalized in forward(). rho has trace 1.
        # So P should be <= 1.
        # And since rho is PSD, P should be >= 0.
        
        self.assertTrue((output >= -1e-5).all())
        self.assertTrue((output <= 1.0 + 1e-5).all())

    def test_backward_pass(self):
        print("\n=== Testing Backward Pass ===")
        embedding = DensityMatrixEmbedding(self.embed_dim).to(self.device)
        evolution = LindbladEvolution(self.embed_dim).to(self.device)
        measurement = QuantumMeasurement(self.embed_dim).to(self.device)
        
        x = torch.randn(self.batch_size, self.embed_dim, requires_grad=True).to(self.device)
        rho = embedding(x)
        rho = evolution(rho)
        output = measurement(rho)
        
        loss = output.mean()
        loss.backward()
        
        print("Gradient check passed if no errors.")
        self.assertIsNotNone(x.grad)
        print(f"Input gradient norm: {x.grad.norm().item()}")

if __name__ == '__main__':
    unittest.main()
