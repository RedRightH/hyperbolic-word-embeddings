"""
Installation and environment test script.

Run this script to verify that all dependencies are installed correctly
and the project structure is set up properly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('gensim', 'Gensim'),
        ('nltk', 'NLTK'),
        ('networkx', 'NetworkX'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            failed.append(name)
    
    # Test geoopt separately (optional but recommended)
    try:
        import geoopt
        print(f"  ✓ Geoopt (Riemannian optimization)")
    except ImportError:
        print(f"  ⚠ Geoopt - NOT INSTALLED (optional, will use fallback)")
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed!")
    return True

def test_project_structure():
    """Test that project directories exist."""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'src/preprocessing',
        'src/models',
        'src/training',
        'src/evaluation',
        'src/visualization',
        'src/utils',
        'notebooks',
        'results',
        'results/figures',
        'results/tables',
        'results/trained_models',
        'experiments'
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            missing.append(dir_path)
    
    if missing:
        print(f"\n⚠ Missing directories (will be created automatically): {len(missing)}")
        return True  # Not critical, will be created
    
    print("\n✅ Project structure verified!")
    return True

def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        return False
    
    print("  ✓ Python version OK")
    return True

def test_nltk_data():
    """Test if NLTK WordNet data is available."""
    print("\nTesting NLTK data...")
    
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
            print("  ✓ WordNet data available")
            return True
        except LookupError:
            print("  ⚠ WordNet data not found")
            print("  Run: python data/download_wordnet.py")
            return False
    except ImportError:
        print("  ✗ NLTK not installed")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA (GPU support)...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    PyTorch version: {torch.__version__}")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU)")
            print(f"    PyTorch version: {torch.__version__}")
            return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("HYPERBOLIC VS EUCLIDEAN EMBEDDINGS - INSTALLATION TEST")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Python Version", test_python_version()))
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("NLTK Data", test_nltk_data()))
    results.append(("CUDA Support", test_cuda()))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    all_critical_passed = all([
        results[0][1],  # Python version
        results[1][1],  # Packages
        results[2][1],  # Structure
    ])
    
    print("\n" + "=" * 70)
    
    if all_critical_passed:
        print("✅ INSTALLATION VERIFIED!")
        print("\nYou're ready to run the project!")
        print("\nNext steps:")
        print("  1. Download WordNet: python data/download_wordnet.py")
        print("  2. Run quick demo: python quick_demo.py")
        print("  3. Or run full pipeline: python run_full_pipeline.py")
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print("\nPlease fix the issues above and run this test again.")
        print("Install dependencies: pip install -r requirements.txt")
    
    print("=" * 70)
    
    return all_critical_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
