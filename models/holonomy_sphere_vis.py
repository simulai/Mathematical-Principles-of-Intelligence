"""
Cognitive Holonomy - Geometric Simulation
-----------------------------------------
This script visualizes the concept of "Holonomy" on a sphere, which serves as the 
theoretical basis for the "Cognitive Holonomy" (formerly Zhang Invariant) in our cognitive architecture.

DISCLAIMER:
The "Cognitive Holonomy" is a project-specific term used to describe the geometric goal 
of minimizing logical curvature (holonomy) in reasoning paths. In our neural network 
implementations (e.g., HolonomyLoss), we approximate this via **Discrete Geodesic Curvature**,
minimizing the angle between consecutive semantic velocity vectors to approximate 
parallel transport.

This simulation demonstrates the exact geometric phenomenon: parallel transport 
around a closed loop resulting in vector rotation (non-zero holonomy).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_sphere_coordinates(radius=1.0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def plot_vector(ax, origin, vector, color='r', label=None):
    ax.quiver(origin[0], origin[1], origin[2], 
              vector[0], vector[1], vector[2], 
              color=color, length=0.3, normalize=True, label=label)

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Sphere
    x, y, z = get_sphere_coordinates(0.98)
    ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='none')

    # Define Points
    A = np.array([1, 0, 0]) # Start
    B = np.array([0, 1, 0])
    C = np.array([0, 0, 1]) # North Pole
    
    # Define Path (Geodesics)
    # Path 1: A -> B (Equator)
    t = np.linspace(0, np.pi/2, 20)
    path1_x = np.cos(t)
    path1_y = np.sin(t)
    path1_z = np.zeros_like(t)
    ax.plot(path1_x, path1_y, path1_z, 'k-', linewidth=2)

    # Path 2: B -> C (Longitude 90)
    t = np.linspace(0, np.pi/2, 20)
    path2_x = np.zeros_like(t)
    path2_y = np.cos(t)
    path2_z = np.sin(t)
    ax.plot(path2_x, path2_y, path2_z, 'k-', linewidth=2)
    
    # Path 3: C -> A (Prime Meridian)
    t = np.linspace(0, np.pi/2, 20)
    path3_x = np.sin(t)
    path3_y = np.zeros_like(t)
    path3_z = np.cos(t)
    ax.plot(path3_x, path3_y, path3_z, 'k-', linewidth=2)

    # Parallel Transport Simulation
    # Initial Vector at A: Pointing North (tangent to sphere at A)
    v_start = np.array([0, 0, 1])
    plot_vector(ax, A, v_start, color='g', label='Start Vector')
    
    # Transport A -> B (Along equator)
    # Since we move along equator and vector is perpendicular (North), it stays North.
    v_at_B = np.array([0, 0, 1])
    # However, at B (0,1,0), North is (0,0,1). It is still tangent.
    # But wait, let's look at the angle relative to the path.
    # At A, path is (0,1,0), vector is (0,0,1). Angle is 90.
    # At B, path incoming is (1,0,0) (wait, tangent of circle at pi/2 is (-1,0,0)).
    # Let's be precise.
    # Tangent of path 1 at t: (-sin t, cos t, 0).
    # At t=0 (A): (0, 1, 0). Vector: (0, 0, 1). Cross product (normal) is (1,0,0).
    # At t=pi/2 (B): (-1, 0, 0). Vector: (0, 0, 1).
    # The vector (0,0,1) is parallel transported along equator.
    plot_vector(ax, B, v_at_B, color='y', label='Transported to B')

    # Transport B -> C (Along longitude)
    # Path 2 tangent: (0, -sin t, cos t).
    # At B (t=0): (0, 0, 1). 
    # Current vector v_at_B is (0, 0, 1).
    # So the vector is TANGENT to the path 2.
    # Parallel transport of a tangent vector to a geodesic remains tangent.
    # So at C (t=pi/2), tangent is (0, -1, 0).
    # So v_at_C should be (0, -1, 0).
    v_at_C = np.array([0, -1, 0])
    plot_vector(ax, C, v_at_C, color='orange', label='Transported to C')

    # Transport C -> A (Along prime meridian)
    # Path 3 tangent: (cos t, 0, -sin t).
    # At C (t=0): (1, 0, 0).
    # Current vector v_at_C is (0, -1, 0).
    # The vector is perpendicular to the path tangent (1,0,0) and normal (0,0,1).
    # Wait, at C (0,0,1), normal is (0,0,1).
    # Tangent is (1,0,0).
    # Vector is (0,-1,0).
    # This vector is perpendicular to tangent.
    # So it should stay perpendicular to tangent as we move.
    # At A (t=pi/2): Point (1,0,0). Normal (1,0,0).
    # Tangent of path is (0, 0, -1).
    # Vector must be perpendicular to tangent (0,0,-1) and normal (1,0,0).
    # So vector must be (0, 1, 0) or (0, -1, 0).
    # Let's trace carefully.
    # At C, vector is (0, -1, 0) (pointing "West" in global coords, or "Right" if facing A).
    # As we move to A, we go "Down".
    # The vector (0, -1, 0) is constant in Euclidean space? No, must be tangent to surface.
    # (0, -1, 0) is always tangent to the prime meridian arc?
    # At any point on Path 3: (sin t, 0, cos t). Normal is (sin t, 0, cos t).
    # Vector (0, -1, 0) has dot product 0 with normal. So it is always tangent.
    # So the vector stays (0, -1, 0) in Euclidean coordinates!
    v_end = np.array([0, -1, 0])
    plot_vector(ax, A, v_end, color='r', label='End Vector')

    # Result:
    # Start Vector at A: (0, 0, 1)
    # End Vector at A: (0, -1, 0)
    # They are 90 degrees apart.
    
    # Annotations
    ax.text(1.1, 0, 0, "A", color='k')
    ax.text(0, 1.1, 0, "B", color='k')
    ax.text(0, 0, 1.1, "C", color='k')
    
    ax.set_title("Cognitive Holonomy Simulation: Holonomy on a Sphere\nCognitive Manifold Curvature", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("H")
    
    # Legend
    # Create proxy artists for legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='End Vector (Rotated)')
    green_patch = mpatches.Patch(color='green', label='Start Vector')
    plt.legend(handles=[green_patch, red_patch])
    
    output_path = 'cognitive_holonomy_simulation.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    print("Cognitive Holonomy detected: Vector rotated by 90 degrees.")
    print("This represents 'Confusion' or energy dissipation in the cognitive loop.")
    print("In a 'Superconductive' (Flat) Cognitive Manifold, the vector would return unchanged.")
    # plt.show()

if __name__ == "__main__":
    main()