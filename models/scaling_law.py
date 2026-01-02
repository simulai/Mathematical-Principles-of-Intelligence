import numpy as np
import matplotlib.pyplot as plt

def efficiency_psi(b):
    """
    Calculates the informational efficiency Psi(b) = ln(b) / b.
    """
    # Avoid division by zero if b includes 0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(b) / b

def main():
    # Define range for b
    b_values = np.linspace(0.1, 10, 500)
    psi_values = efficiency_psi(b_values)

    # Theoretical maximum
    e = np.e
    psi_max = efficiency_psi(e)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(b_values, psi_values, label=r'$\Psi(b) = \frac{\ln b}{b}$', color='blue', linewidth=2)
    
    # Mark maximum
    plt.scatter([e], [psi_max], color='red', zorder=5)
    plt.axvline(x=e, color='red', linestyle='--', alpha=0.5, label=f'Optimal $b = e \\approx {e:.3f}$')
    
    # Mark integers
    integers = [2, 3, 4]
    for i in integers:
        psi_i = efficiency_psi(i)
        plt.scatter([i], [psi_i], color='green', zorder=5)
        plt.annotate(f'b={i}\n{psi_i:.3f}', (i, psi_i), xytext=(0, 10), textcoords='offset points', ha='center')

    # Annotate maximum
    plt.annotate(f'Max\n{psi_max:.3f}', (e, psi_max), xytext=(0, 10), textcoords='offset points', ha='center', color='red')

    # Labels and styling
    plt.title('Cognitive Efficiency Scaling Law: The e-base Optimum', fontsize=14)
    plt.xlabel('Branching Factor (b)', fontsize=12)
    plt.ylabel('Efficiency $\Psi(b)$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(bottom=0)

    # Save and show
    output_path = 'psi_curve_simulation.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
