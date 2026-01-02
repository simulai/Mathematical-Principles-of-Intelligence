import torch
import torch.nn.functional as F
import numpy as np

def compute_stable_rank(embeddings: torch.Tensor, epsilon=1e-6) -> float:
    """
    Computes the Stable Rank of a batch of embeddings.
    Stable Rank is a proxy for the effective dimensionality of the data manifold.
    
    Args:
        embeddings: (Batch, Dim) tensor
    Returns:
        float: Stable Rank value (1 <= r <= Dim)
    """
    if embeddings.dim() != 2:
        return 0.0
        
    # Center the embeddings
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # Compute SVD
    # Note: For large batches, consider using randomized SVD or just covariance
    try:
        _, S, _ = torch.svd(centered)
    except RuntimeError:
        # Fallback for numerical instability
        return 0.0
        
    # Stable Rank = sum(sigma^2) / max(sigma^2)
    # This is equivalent to ||X||_F^2 / ||X||_2^2
    eigenvalues = S ** 2
    max_eigenval = eigenvalues.max() + epsilon
    sum_eigenval = eigenvalues.sum()
    
    return (sum_eigenval / max_eigenval).item()

def compute_spectral_entropy(embeddings: torch.Tensor, epsilon=1e-6) -> float:
    """
    Computes the Shannon Entropy of the singular value spectrum (Spectral Entropy).
    High entropy -> Energy spread across many dimensions (White Noise / High Capacity).
    Low entropy -> Energy concentrated in few dimensions (Collapse / Low Rank).
    
    Args:
        embeddings: (Batch, Dim) tensor
    Returns:
        float: Normalized Spectral Entropy (0 <= H <= 1)
    """
    if embeddings.dim() != 2:
        return 0.0
        
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    try:
        _, S, _ = torch.svd(centered)
    except RuntimeError:
        return 0.0
        
    # Normalize singular values to form a probability distribution
    S_sum = S.sum() + epsilon
    probs = S / S_sum
    
    # Compute Entropy
    entropy = -torch.sum(probs * torch.log(probs + epsilon))
    
    # Normalize by log(min(Batch, Dim)) to get range [0, 1]
    max_entropy = np.log(min(embeddings.shape))
    
    return (entropy / max_entropy).item()

def compute_erpm_proxy(embeddings: torch.Tensor) -> dict:
    """
    Computes a set of manifold metrics serving as a proxy for ERPM.
    """
    return {
        "stable_rank": compute_stable_rank(embeddings),
        "spectral_entropy": compute_spectral_entropy(embeddings)
    }
