import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def sinkhorn_projection(matrix, iterations=10):
    """
    Simulates the core of mHC in the paper: projecting the matrix onto the doubly stochastic manifold (Birkhoff Polytope).
    This can be seen as an operator to maintain the Zhang Invariant.
    """
    # Pre-processing: exp to ensure non-negative, similar to softmax logic in attention or gating
    M = np.exp(matrix) 
    for _ in range(iterations):
        # Row normalization
        row_sum = M.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sum[row_sum == 0] = 1e-10
        M /= row_sum
        
        # Column normalization
        col_sum = M.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1e-10
        M /= col_sum
    return M

def simulate_propagation(depth=100, constrained=True, expansion_rate=4, dim=64):
    """
    Simulates signal propagation through layers.
    """
    # Initial signal (feature vector)
    # Using normal distribution, then normalizing to norm 1 for clarity
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    
    history = [np.linalg.norm(x)]
    
    # Current signal
    curr_x = x.copy()
    
    # Expansion dimension (simulating Hyper-Connections width)
    hidden_dim = dim * expansion_rate
    
    for _ in range(depth):
        # Simulate layer transformation matrix
        # In HC, this is H_res (n*n matrix acting on residual stream)
        # We simulate the transformation on the hidden dimension
        
        # 1. Expand to hidden dim (simulating H_pre or just mapping)
        # For simplicity, we just use a square matrix transition on the 'residual stream'
        # Assuming we are already IN the residual stream.
        
        # Random initialization (unconstrained)
        # Using Xavier/Glorot initialization scale
        scale = np.sqrt(2.0 / (dim + dim))
        W = np.random.randn(dim, dim) * scale
        
        if constrained:
            # Apply manifold constraint (mHC core logic)
            # The paper uses Sinkhorn-Knopp on H_res
            W = sinkhorn_projection(W)
        else:
            # For unconstrained, we might add some noise or shift to simulate
            # the "drift" that happens in deep networks without normalization
            # The issue in HC is that singular values drift away from 1.
            # Standard random matrix tends to be okay-ish for short depth, 
            # but let's make it slightly non-unitary to show explosion/vanishing
            pass

        # 2. Apply transformation
        curr_x = np.dot(W, curr_x)
        
        # 3. Store norm
        history.append(np.linalg.norm(curr_x))
        
    return np.array(history)

def main():
    # Parameters
    depth = 50
    num_trials = 20 # Run multiple trials to show variance
    
    # Data collection
    unconstrained_data = []
    constrained_data = []
    
    for _ in range(num_trials):
        unconstrained_data.append(simulate_propagation(depth=depth, constrained=False))
        constrained_data.append(simulate_propagation(depth=depth, constrained=True))
        
    unconstrained_data = np.array(unconstrained_data)
    constrained_data = np.array(constrained_data)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, depth)
    # Set y-limit dynamically or fixed? Fixed is better to show explosion
    ax.set_ylim(0, 5) 
    
    ax.set_title("Cognitive Signal Stability: Manifold vs. Unconstrained Flow", fontsize=14)
    ax.set_xlabel("Layer Depth (Time)", fontsize=12)
    ax.set_ylabel("Signal Norm (Information Consistency)", fontsize=12)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Ideal Stability (Zhang Invariant)")
    
    # Lines
    lines_unconstrained = []
    lines_constrained = []
    
    # Create line objects
    for _ in range(num_trials):
        l_u, = ax.plot([], [], color='red', alpha=0.1)
        lines_unconstrained.append(l_u)
        l_c, = ax.plot([], [], color='green', alpha=0.3) # Higher alpha for constrained as they overlap
        lines_constrained.append(l_c)
        
    # Mean lines
    mean_line_u, = ax.plot([], [], color='darkred', linewidth=2, label="Unconstrained (Chaos)")
    mean_line_c, = ax.plot([], [], color='darkgreen', linewidth=2, label="Manifold Constrained (mHC)")
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # Text annotation
    text_iter = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        for l in lines_unconstrained + lines_constrained:
            l.set_data([], [])
        mean_line_u.set_data([], [])
        mean_line_c.set_data([], [])
        text_iter.set_text('')
        return lines_unconstrained + lines_constrained + [mean_line_u, mean_line_c, text_iter]

    def update(frame):
        # Frame goes from 0 to depth
        x_data = np.arange(frame + 1)
        
        # Update individual lines
        for i in range(num_trials):
            lines_unconstrained[i].set_data(x_data, unconstrained_data[i, :frame+1])
            lines_constrained[i].set_data(x_data, constrained_data[i, :frame+1])
            
        # Update mean lines
        mean_u = np.mean(unconstrained_data[:, :frame+1], axis=0)
        mean_c = np.mean(constrained_data[:, :frame+1], axis=0)
        
        mean_line_u.set_data(x_data, mean_u)
        mean_line_c.set_data(x_data, mean_c)
        
        text_iter.set_text(f'Layer: {frame}/{depth}')
        
        return lines_unconstrained + lines_constrained + [mean_line_u, mean_line_c, text_iter]

    # Create Animation
    ani = animation.FuncAnimation(fig, update, frames=depth, init_func=init, blit=True, interval=100)
    
    # Save
    output_path = 'manifold_alignment.gif'
    # Use Pillow writer for GIF
    ani.save(output_path, writer='pillow', fps=15)
    print(f"Animation saved to {output_path}")
    
    # Also save a static final frame
    update(depth-1)
    plt.savefig('manifold_alignment_static.png')
    print(f"Static plot saved to manifold_alignment_static.png")

if __name__ == "__main__":
    main()
