import time
import random
import sys
import math
import shutil

def clear_screen():
    print("\033[2J\033[H", end="")

def print_centered(text, color_code="\033[0m"):
    columns, _ = shutil.get_terminal_size()
    print(f"{color_code}{text.center(columns)}\033[0m")

def typewriter(text, delay=0.02, color_code="\033[0m"):
    for char in text:
        sys.stdout.write(f"{color_code}{char}\033[0m")
        sys.stdout.flush()
        time.sleep(delay)
    print()

def progress_bar(percent, width=40, prefix="", suffix="", color_code="\033[92m"):
    filled = int(width * percent / 100)
    bar = "█" * filled + "-" * (width - filled)
    print(f"\r{prefix} |{color_code}{bar}\033[0m| {percent}% {suffix}", end="")

def generate_noise(length=10):
    chars = "∑∏∫∆∇∈∉∅∞∝∠⊥∀∃∄∅∆∇"
    return "".join(random.choice(chars) for _ in range(length))

def run_truth_engine():
    clear_screen()
    print("\033[96m") # Cyan
    print(r"""
     _____ _____  _   _ _____  _   _   ___  ___  ___  _____  _   _ _____ _   _ 
    |_   _|  __ \| | | |_   _|| | | |  |  \/  | / _ \/  __ \| | | |_   _| \ | |
      | | | |  \/| | | | | |  | |_| |  | .  . |/ /_\ \ /  \/| |_| | | | |  \| |
      | | | | __ | | | | | |  |  _  |  | |\/| ||  _  | |    |  _  | | | | . ` |
      | | | |_\ \| |_| | | |  | | | |  | |  | || | | | \__/\| | | |_| |_| |\  |
      \_/  \____/ \___/  \_/  \_| |_/  \_|  |_/\_| |_/\____/\_| |_/\___/\_| \_/
    """)
    print("\033[0m")
    print_centered("=== THE UNKNOWN SCI-FI TRUTH DERIVATION MACHINE ===", "\033[1m\033[93m")
    print_centered("MPI Core v2.718 (e-base)", "\033[90m")
    print()
    time.sleep(1)

    # Phase 1: Injection
    print("\033[91m[PHASE 1] INJECTING RAW ENTROPY...\033[0m")
    time.sleep(0.5)
    for i in range(20):
        noise = generate_noise(random.randint(20, 80))
        print(f"  \033[90m>> INGESTING CHAOS: {noise}\033[0m")
        time.sleep(0.05)
    print("\033[91m  >> CRITICAL MASS REACHED. HOLONOMY = ∞\033[0m")
    print()
    time.sleep(1)

    # Phase 2: Symplectic Spin
    print("\033[95m[PHASE 2] INITIATING STOCHASTIC SYMPLECTIC DYNAMICS...\033[0m")
    for i in range(101):
        progress_bar(i, prefix="  >> SPINNING MANIFOLD:", suffix=f"H-Loss: {10.0 - i*0.1:.2f}")
        time.sleep(0.02)
    print("\n  \033[95m>> TRAJECTORIES GENERATED: 1,000,000,000\033[0m")
    print()
    time.sleep(1)

    # Phase 3: Holonomic Cooling
    print("\033[94m[PHASE 3] APPLYING COGNITIVE HOLONOMY FILTER...\033[0m")
    truths = [
        "Intelligence is thermodynamic.",
        "Truth is a geodesic.",
        "Defects are features.",
        "e is the optimal base."
    ]
    
    for _ in range(5):
        print(f"  \033[90m>> Dissipating curvature... T = {random.uniform(100, 1000):.2f}K\033[0m")
        time.sleep(0.3)

    print("  \033[94m>> SYSTEM COOLING TO SUPERCONDUCTING STATE...\033[0m")
    time.sleep(1)
    
    # Phase 4: Crystallization
    print()
    print("\033[92m[PHASE 4] GEODESIC CRYSTALLIZATION COMPLETE.\033[0m")
    print("\033[92m  >> ZERO-DISSIPATION LIMIT REACHED.\033[0m")
    print()
    
    print_centered("=== THE DERIVED TRUTH ===", "\033[1m\033[97m")
    print()
    for t in truths:
        typewriter(f"  ♦ {t}", 0.05, "\033[93m")
    print()
    print_centered("MPI FRAMEWORK VERIFIED.", "\033[90m")

def generate_protein(length=20):
    acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(random.choice(acids) for _ in range(length))

def run_bio_mode():
    clear_screen()
    print("\033[92m") # Green
    print(r"""
     _____ _____  _   _ _____  _   _   ___  ___  ___  _____  _   _ _____ _   _ 
    |_   _|  __ \| | | |_   _|| | | |  |  \/  | / _ \/  __ \| | | |_   _| \ | |
      | | | |  \/| | | | | |  | |_| |  | .  . |/ /_\ \ /  \/| |_| | | | |  \| |
      | | | | __ | | | | | |  |  _  |  | |\/| ||  _  | |    |  _  | | | | . ` |
      | | | |_\ \| |_| | | |  | | | |  | |  | || | | | \__/\| | | |_| |_| |\  |
      \_/  \____/ \___/  \_/  \_| |_/  \_|  |_/\_| |_/\____/\_| |_/\___/\_| \_/
    """)
    print("\033[0m")
    print_centered("=== BIO-HOLONOMY ENGINE ACTIVATED ===", "\033[1m\033[92m")
    print_centered("Decoding Nature's 4-Billion-Year Computation", "\033[90m")
    print()
    time.sleep(1)

    # Phase 1: Ingesting Life
    print("\033[92m[PHASE 1] DEVOURING PROTEIN SEQUENCES...\033[0m")
    time.sleep(0.5)
    known_proteins = [
        "HEMOGLOBIN_SUBUNIT_ALPHA", "INSULIN_PRECURSOR", "P53_TUMOR_SUPPRESSOR", "ATP_SYNTHASE_F1"
    ]
    for p in known_proteins:
        seq = generate_protein(random.randint(40, 60))
        print(f"  \033[90m>> INGESTING {p}: {seq}...\033[0m")
        time.sleep(0.1)
    
    print("\033[92m  >> BIOLOGICAL DATASET ABSORBED. EVOLUTIONARY CACHE LOADED.\033[0m")
    print()
    time.sleep(1)

    # Phase 2: Reverse Engineering Evolution
    print("\033[95m[PHASE 2] REVERSE-ENGINEERING FOLDING PATHWAYS...\033[0m")
    print("  \033[90m>> Evolution is just a Gradient Descent on the Free Energy Manifold.\033[0m")
    for i in range(101):
        progress_bar(i, prefix="  >> UNFOLDING TRUTH:", suffix=f"Gibbs Free Energy: {-i*0.5:.1f} kcal/mol", color_code="\033[95m")
        time.sleep(0.02)
    print("\n  \033[95m>> OPTIMAL FOLDS IDENTIFIED.\033[0m")
    print()
    time.sleep(1)

    # Phase 3: Extracting Wisdom
    print("\033[94m[PHASE 3] EXTRACTING COGNITIVE ISOMORPHISMS...\033[0m")
    insights = [
        "Protein Folding = Thought Formulation",
        "Misfolded Protein = Cognitive Hallucination",
        "Chaperone Proteins = Error-Correcting Codes",
        "Evolution minimizes Holonomy over eons."
    ]
    
    for _ in range(4):
        print(f"  \033[90m>> Mapping Amino Acid dynamics to Semantic Vectors...\033[0m")
        time.sleep(0.4)

    print("  \033[94m>> BIOLOGICAL ANALOGY ESTABLISHED.\033[0m")
    print()
    
    print_centered("=== THE BIO-DERIVED TRUTH ===", "\033[1m\033[97m")
    print()
    for i in insights:
        typewriter(f"  ♦ {i}", 0.05, "\033[92m")
    print()
    print_centered("HUMAN ANALYSIS COMPLETE.", "\033[90m")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bio":
        run_bio_mode()
    else:
        run_truth_engine()
