"""
Full pipeline script to run all experiments end-to-end.

This script:
1. Downloads WordNet data
2. Extracts hypernym relationships
3. Builds hierarchy graph
4. Trains both Euclidean and Poincaré embeddings
5. Evaluates both models
6. Generates visualizations
"""

import sys
from pathlib import Path

def run_pipeline():
    """Run the complete experimental pipeline."""
    
    print("=" * 80)
    print("HYPERBOLIC VS EUCLIDEAN EMBEDDINGS - FULL PIPELINE")
    print("=" * 80)
    print()
    
    steps = [
        ("Downloading WordNet data", "python data/download_wordnet.py"),
        ("Extracting hypernym relationships", "python src/preprocessing/extract_wordnet.py"),
        ("Building hierarchy graph", "python src/preprocessing/build_hierarchy.py"),
        ("Training Euclidean embeddings", "python src/training/train_euclidean.py"),
        ("Training Poincaré embeddings", "python src/training/train_hyperbolic.py"),
        ("Evaluating reconstruction error", "python src/evaluation/reconstruction_error.py"),
        ("Evaluating link prediction", "python src/evaluation/link_prediction.py"),
        ("Visualizing Euclidean embeddings", "python src/visualization/plot_euclidean.py"),
        ("Visualizing Poincaré embeddings", "python src/visualization/plot_poincare_disk.py"),
    ]
    
    import subprocess
    
    for i, (description, command) in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {description}")
        print("-" * 80)
        
        try:
            result = subprocess.run(
                command.split(),
                check=True,
                capture_output=False,
                text=True
            )
            print(f"✓ {description} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in {description}")
            print(f"  Command: {command}")
            print(f"  Error: {e}")
            
            user_input = input("\nContinue with remaining steps? (y/n): ")
            if user_input.lower() != 'y':
                print("Pipeline stopped by user.")
                return False
        except FileNotFoundError:
            print(f"✗ Could not find command: {command}")
            print("  Make sure you're running from the project root directory")
            return False
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - Trained models: results/trained_models/")
    print("  - Evaluation metrics: results/tables/")
    print("  - Visualizations: results/figures/")
    print("\nNext steps:")
    print("  - Explore results in Jupyter notebooks: jupyter notebook notebooks/")
    print("  - Check evaluation metrics in results/tables/")
    print("  - View visualizations in results/figures/")
    
    return True

if __name__ == "__main__":
    import os
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    print()
    
    success = run_pipeline()
    
    sys.exit(0 if success else 1)
