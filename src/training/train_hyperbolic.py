import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.poincare_embeddings import PoincareEmbeddings
from src.preprocessing.dataset_utils import load_split, load_mapping
from src.utils.config import MODELS_DIR, HYPERBOLIC_CONFIG
import argparse

def train_hyperbolic_embeddings(embedding_dim=None, epochs=None, learning_rate=None, dataset_prefix=None):
    """
    Train Poincaré (hyperbolic) embeddings on WordNet data.
    
    Args:
        embedding_dim: dimension of embeddings
        epochs: number of training epochs
        learning_rate: learning rate
    """
    print("=" * 60)
    print("Training Poincaré (Hyperbolic) Embeddings")
    print("=" * 60)
    
    try:
        if dataset_prefix:
            train_edges, test_edges = load_split(
                train_file=f"{dataset_prefix}_train_edges.pkl",
                test_file=f"{dataset_prefix}_test_edges.pkl",
            )
            node2id, id2node = load_mapping(
                node2id_file=f"{dataset_prefix}_node2id.pkl",
                id2node_file=f"{dataset_prefix}_id2node.pkl",
            )
        else:
            train_edges, test_edges = load_split()
            node2id, id2node = load_mapping()
    except FileNotFoundError:
        print("Training data not found. Running data preparation...")
        from src.training.trainer import prepare_training_data
        train_edges, test_edges, node2id, id2node = prepare_training_data(dataset_prefix=dataset_prefix)
    
    kwargs = {}
    if embedding_dim:
        kwargs['embedding_dim'] = embedding_dim
    if epochs:
        kwargs['epochs'] = epochs
    if learning_rate:
        kwargs['learning_rate'] = learning_rate
    
    model = PoincareEmbeddings(**kwargs)
    
    model.train(train_edges, node2id, id2node)
    
    model_filename = f"{dataset_prefix}_poincare_embeddings.pkl" if dataset_prefix else "poincare_embeddings.pkl"
    model_path = MODELS_DIR / model_filename
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Poincaré embeddings')
    parser.add_argument('--dim', type=int, default=None, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--dataset-prefix', type=str, default=None, help='Dataset prefix (uses prefixed processed artifacts)')
    
    args = parser.parse_args()
    
    train_hyperbolic_embeddings(
        embedding_dim=args.dim, 
        epochs=args.epochs,
        learning_rate=args.lr,
        dataset_prefix=args.dataset_prefix
    )
