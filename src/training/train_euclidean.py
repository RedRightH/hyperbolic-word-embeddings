import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.euclidean_embeddings import EuclideanEmbeddings
from src.preprocessing.dataset_utils import load_split, load_mapping
from src.utils.config import MODELS_DIR, EUCLIDEAN_CONFIG
import argparse

def train_euclidean_embeddings(embedding_dim=None, epochs=None):
    """
    Train Euclidean embeddings on WordNet data.
    
    Args:
        embedding_dim: dimension of embeddings
        epochs: number of training epochs
    """
    print("=" * 60)
    print("Training Euclidean Embeddings")
    print("=" * 60)
    
    try:
        train_edges, test_edges = load_split()
        node2id, id2node = load_mapping()
    except FileNotFoundError:
        print("Training data not found. Running data preparation...")
        from src.training.trainer import prepare_training_data
        train_edges, test_edges, node2id, id2node = prepare_training_data()
    
    kwargs = {}
    if embedding_dim:
        kwargs['embedding_dim'] = embedding_dim
    if epochs:
        kwargs['epochs'] = epochs
    
    model = EuclideanEmbeddings(**kwargs)
    
    model.train(train_edges, node2id, id2node)
    
    model_path = MODELS_DIR / "euclidean_embeddings.pkl"
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Euclidean embeddings')
    parser.add_argument('--dim', type=int, default=None, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    
    args = parser.parse_args()
    
    train_euclidean_embeddings(embedding_dim=args.dim, epochs=args.epochs)
