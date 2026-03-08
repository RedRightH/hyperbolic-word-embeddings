# Getting Started Guide

Welcome to the Hyperbolic vs Euclidean Word Embeddings project! This guide will help you get up and running quickly.

## Quick Start (5 minutes)

### Option 1: Quick Demo

Run a fast demo with a small dataset:

```bash
python quick_demo.py
```

This will:
- Extract 1000 WordNet edges
- Train both models with reduced parameters
- Show quick comparison results

### Option 2: Full Pipeline

Run the complete experiment:

```bash
python run_full_pipeline.py
```

This will execute all steps automatically (takes ~30-60 minutes depending on your hardware).

## Step-by-Step Guide

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WordNet
python data/download_wordnet.py
```

### 2. Data Preparation

```bash
# Extract hypernym relationships from WordNet
python src/preprocessing/extract_wordnet.py

# Build graph structure and compute distances
python src/preprocessing/build_hierarchy.py
```

**Output:**
- `data/raw/wordnet_edges.txt` - Edge list
- `data/processed/wordnet_graph.pkl` - NetworkX graph
- `data/processed/tree_distances.pkl` - Pairwise distances

### 3. Train Models

**Euclidean embeddings (Word2Vec):**
```bash
python src/training/train_euclidean.py
```

**Poincaré embeddings:**
```bash
python src/training/train_hyperbolic.py
```

**Custom parameters:**
```bash
# Euclidean with 128 dimensions
python src/training/train_euclidean.py --dim 128 --epochs 100

# Poincaré with 5 dimensions
python src/training/train_hyperbolic.py --dim 5 --epochs 300 --lr 0.1
```

**Output:**
- `results/trained_models/euclidean_embeddings.pkl`
- `results/trained_models/poincare_embeddings.pkl`

### 4. Evaluate Models

**Reconstruction error:**
```bash
python src/evaluation/reconstruction_error.py
```

**Link prediction:**
```bash
python src/evaluation/link_prediction.py
```

**Output:**
- `results/tables/reconstruction_error.json`
- `results/tables/link_prediction.json`

### 5. Generate Validation Visualizations

**All validation visualizations:**
```bash
python src/visualization/validate_hierarchy.py
```

**Normalized separation analysis:**
```bash
python src/visualization/validate_separation.py
```

**Output:**
- `results/figures/validation_euclidean_correlation.png`
- `results/figures/validation_poincare_correlation.png`
- `results/figures/validation_euclidean_parent_child.png`
- `results/figures/validation_poincare_parent_child.png`
- `results/figures/validation_euclidean_depth_norm.png`
- `results/figures/validation_poincare_depth_norm.png`
- `results/figures/validation_comparison_summary.png`
- `results/figures/validation_normalized_separation.png`

These visualizations prove that Poincaré embeddings preserve hierarchical structure better than Euclidean embeddings.

## Understanding the Results

### Reconstruction Error

Lower values = better preservation of graph structure

```json
{
  "euclidean": {
    "mse": 49.62,
    "mae": 6.06,
    "rmse": 7.04,
    "correlation": -0.08
  },
  "poincare": {
    "mse": 43.89,
    "mae": 5.54,
    "rmse": 6.62,
    "correlation": -0.23
  }
}
```

**Interpretation:** Poincaré embeddings better preserve hierarchical distances.

### Link Prediction

Higher Hits@K = better prediction accuracy

```json
{
  "euclidean": {
    "mean_rank": 838.78,
    "median_rank": 965.0,
    "hits@1": 0.0,
    "hits@5": 0.0,
    "hits@10": 0.005
  },
  "poincare": {
    "mean_rank": 350.98,
    "median_rank": 297.5,
    "hits@1": 0.0,
    "hits@5": 0.0,
    "hits@10": 0.005
  }
}
```

**Interpretation:** Poincaré embeddings predict hypernym relationships more accurately.

## Configuration

Edit `src/utils/config.py` to customize:

```python
EUCLIDEAN_CONFIG = {
    'embedding_dim': 100,  # Embedding dimension
    'epochs': 50,          # Training epochs
    'learning_rate': 0.025 # Learning rate
}

HYPERBOLIC_CONFIG = {
    'embedding_dim': 10,   # Embedding dimension (much smaller!)
    'epochs': 300,         # Training epochs
    'learning_rate': 0.1,  # Learning rate
    'batch_size': 32       # Batch size
}
```

## Troubleshooting

### Issue: "NLTK WordNet not found"

**Solution:**
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Issue: Training is slow

**Solutions:**
- Use GPU if available (PyTorch will auto-detect)
- Reduce dataset size in `extract_wordnet.py` (set `limit=5000`)
- Reduce batch size in config
- Reduce number of epochs

### Issue: Out of memory

**Solutions:**
- Reduce batch size: `--batch_size 16`
- Reduce embedding dimension
- Use CPU instead of GPU
- Reduce dataset size

### Issue: Poor results

**Solutions:**
- Increase training epochs
- Adjust learning rate
- Use larger dataset
- Check data quality by examining WordNet edges

## Next Steps

1. **Experiment with parameters**: Try different dimensions and learning rates
2. **Add more data**: Increase WordNet subset size
3. **Compare metrics**: Analyze which model performs better on your data
4. **Visualize**: Generate validation visualizations to analyze results
5. **Extend**: Try other hierarchical datasets (taxonomy, citations, etc.)

## Project Structure Overview

```
├── data/                    # Data files
├── src/                     # Source code
│   ├── preprocessing/       # Data preparation
│   ├── models/             # Embedding models
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation metrics
│   ├── visualization/      # Validation scripts
│   └── utils/              # Utilities
├── results/                # Output files
│   ├── trained_models/     # Saved models
│   ├── tables/             # Evaluation results
│   └── figures/            # Visualizations
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
└── quick_demo.py          # Quick start script
```

## Tips for Best Results

1. **Start small**: Use `quick_demo.py` to verify everything works
2. **Monitor training**: Watch loss values - they should decrease
3. **Compare fairly**: Use same dataset for both models
4. **Visualize early**: Check embeddings during training
5. **Document experiments**: Keep notes on parameters and results

## Common Workflows

### Workflow 1: Quick Experiment
```bash
python quick_demo.py
python src/visualization/validate_hierarchy.py
```

### Workflow 2: Full Evaluation
```bash
python run_full_pipeline.py
# Review results in results/tables/ and results/figures/
```

### Workflow 3: Parameter Tuning
```bash
# Train with different parameters
python src/training/train_hyperbolic.py --dim 5 --epochs 500
python src/training/train_hyperbolic.py --dim 10 --epochs 300
python src/training/train_hyperbolic.py --dim 20 --epochs 200

# Compare results
python src/evaluation/link_prediction.py
```

## Resources

- **Paper**: [Poincaré Embeddings (Nickel & Kiela, 2017)](https://arxiv.org/abs/1705.08039)
- **WordNet**: [Princeton WordNet](https://wordnet.princeton.edu/)
- **Hyperbolic Geometry**: [Wikipedia - Poincaré Disk](https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model)

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review the main README.md
3. Examine error messages carefully
4. Run `quick_demo.py` to isolate the problem

Happy experimenting! 🚀
