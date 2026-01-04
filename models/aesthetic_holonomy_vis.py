
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

def generate_aesthetic_landscape():
    print("Generating Entropy Landscape & Holonomic Attractor...")
    
    # 1. Setup the Grid (Cognitive Manifold Projection)
    # X: Logical Progression (Time/Steps)
    # Y: Semantic Deviation (Holonomy/Error)
    x = np.linspace(0, 10, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 2. Define the Potential Energy Surface (The "Truth Valley")
    # V(x, y) = -log(x+1) (Progress gain) + lambda * y^2 (Holonomy Cost)
    # But we want a "Landscape", so let's make it look like a canyon
    # Base terrain: undulating but generally rising "Energy" cost for deviation
    
    # Holonomy Cost (The Canyon Walls)
    # Steepness depends on the "Criticality" (e-base stability)
    # At b=e, the walls are optimal (steep enough to guide, wide enough to explore)
    holonomy_cost = 1.5 * (1 - np.exp(-Y**2)) # Soft potential well
    
    # Entropy/Complexity Gradient (The "Flow" direction)
    # Lower energy forward
    flow_gradient = -0.1 * X 
    
    # Add some "Cognitive Noise" (Terrain Roughness)
    noise = 0.05 * np.sin(3*X) * np.cos(3*Y)
    
    # Total Free Energy Z
    Z = holonomy_cost + flow_gradient + noise
    
    # 3. Create the Visualization
    fig = plt.figure(figsize=(16, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface
    # Use a "Cool-Warm" colormap to indicate Energy (Blue=Low/Stable, Red=High/Unstable)
    surf = ax.plot_surface(X, Y, Z, cmap='magma_r', alpha=0.8, 
                          linewidth=0, antialiased=True, rstride=2, cstride=2)
    
    # 4. Generate Trajectories (The "Paths")
    
    t = np.linspace(0, 10, 200)
    
    # Path A: The "Holonomic Flow" (Golden Truth)
    # Stays close to Y=0 (Geodesic), low energy
    y_gold = 0.1 * np.sin(2*t) * np.exp(-0.2*t) # Damped oscillation
    z_gold = 1.5 * (1 - np.exp(-y_gold**2)) - 0.1 * t + 0.05 * np.sin(3*t) * np.cos(3*y_gold) + 0.1 # Slightly above surface
    ax.plot(t, y_gold, z_gold, color='gold', linewidth=3, label='Holonomic Truth Flow (H -> 0)')
    
    # Path B: "Fantasy with Return" (Controlled Creativity)
    # Deviates but pulled back
    y_fantasy = 1.5 * np.sin(t) * np.exp(-0.1*(t-3)**2) # Big deviation at t=3
    z_fantasy = 1.5 * (1 - np.exp(-y_fantasy**2)) - 0.1 * t + 0.1
    ax.plot(t, y_fantasy, z_fantasy, color='cyan', linewidth=2, linestyle='--', label='Symplectic Fantasy (Exploration)')
    
    # Path C: "Hallucination" (Divergence)
    # Escapes the valley
    y_halluc = 0.5 * t * np.sin(t) # Growing oscillation
    z_halluc = 1.5 * (1 - np.exp(-y_halluc**2)) - 0.1 * t + 0.1
    ax.plot(t, y_halluc, z_halluc, color='red', linewidth=1.5, alpha=0.6, label='Hallucination (High H)')
    
    # 5. Aesthetics & Labels
    ax.set_title("The Entropy Landscape & Holonomic Attractor", fontsize=16, color='white', pad=20)
    ax.set_xlabel('Reasoning Time (Steps)', fontsize=12, color='white')
    ax.set_ylabel('Semantic Deviation ($\mathcal{H}$)', fontsize=12, color='white')
    ax.set_zlabel('Free Energy $F(b)$', fontsize=12, color='white')
    
    # Dark Background for "Sci-Fi" look
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    
    # Remove pane fills
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set axis colors
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Legend
    ax.legend(loc='upper right', facecolor='#222222', edgecolor='white', labelcolor='white')
    
    # View Angle
    ax.view_init(elev=30, azim=-60)
    
    # Save
    output_path = 'docs/entropy_landscape_vis.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='#121212', dpi=150)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    generate_aesthetic_landscape()
