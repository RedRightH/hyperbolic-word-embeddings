import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseEmbeddingModel
from src.utils.config import HYPERBOLIC_CONFIG
from src.utils.hyperbolic_math import poincare_distance, project_to_poincare_ball
from tqdm import tqdm

try:
    from geoopt.optim import RiemannianAdam
    USING_RIEMANNIAN = True
except ImportError:
    RiemannianAdam = optim.Adam
    USING_RIEMANNIAN = False
    import warnings
    warnings.warn("geoopt not available, using standard Adam optimizer. "
                  "For better results, install geoopt: pip install geoopt", 
                  UserWarning)

class PoincareDataset(Dataset):
    """
    Dataset for Poincaré embeddings training.
    """
    
    def __init__(self, edges, node2id, negative_samples=10):
        """
        Initialize dataset.
        
        Args:
            edges: list of (child, parent) tuples
            node2id: dict mapping node names to IDs
            negative_samples: number of negative samples per positive edge
        """
        self.edges = edges
        self.node2id = node2id
        self.negative_samples = negative_samples
        self.num_nodes = len(node2id)
        
        self.edge_indices = []
        for child, parent in edges:
            if child in node2id and parent in node2id:
                child_id = node2id[child]
                parent_id = node2id[parent]
                self.edge_indices.append((child_id, parent_id))
    
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        child_id, parent_id = self.edge_indices[idx]
        
        negatives = []
        for _ in range(self.negative_samples):
            neg_id = np.random.randint(0, self.num_nodes)
            while neg_id == parent_id:
                neg_id = np.random.randint(0, self.num_nodes)
            negatives.append(neg_id)
        
        return child_id, parent_id, negatives

class PoincareModel(nn.Module):
    """
    Poincaré embedding model.
    """
    
    def __init__(self, num_nodes, embedding_dim):
        """
        Initialize Poincaré model.
        
        Args:
            num_nodes: number of nodes
            embedding_dim: dimension of embeddings
        """
        super(PoincareModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)
    
    def forward(self, child_ids, parent_ids):
        """
        Compute Poincaré distances.
        
        Args:
            child_ids: tensor of child node IDs
            parent_ids: tensor of parent node IDs
        
        Returns:
            tensor of distances
        """
        child_emb = self.embeddings(child_ids)
        parent_emb = self.embeddings(parent_ids)
        
        child_emb = project_to_poincare_ball(child_emb)
        parent_emb = project_to_poincare_ball(parent_emb)
        
        distances = poincare_distance(child_emb, parent_emb)
        
        return distances

class PoincareEmbeddings(BaseEmbeddingModel):
    """
    Poincaré embeddings for hierarchical data.
    """
    
    def __init__(self, embedding_dim=None, **kwargs):
        """
        Initialize Poincaré embedding model.
        
        Args:
            embedding_dim: dimension of embeddings
            **kwargs: additional training parameters
        """
        if embedding_dim is None:
            embedding_dim = HYPERBOLIC_CONFIG['embedding_dim']
        
        super().__init__(embedding_dim)
        self.model = None
        self.config = {**HYPERBOLIC_CONFIG, **kwargs}
        if embedding_dim:
            self.config['embedding_dim'] = embedding_dim
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, edges, node2id, id2node, **kwargs):
        """
        Train Poincaré embeddings on edge data.
        
        Args:
            edges: list of (child, parent) tuples
            node2id: dict mapping node names to IDs
            id2node: dict mapping IDs to node names
            **kwargs: additional training parameters
        """
        self.node2id = node2id
        self.id2node = id2node
        
        config = {**self.config, **kwargs}
        
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        num_nodes = len(node2id)
        self.model = PoincareModel(num_nodes, config['embedding_dim']).to(self.device)
        
        dataset = PoincareDataset(edges, node2id, config['negative_samples'])
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        optimizer = RiemannianAdam(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        
        optimizer_type = "RiemannianAdam" if USING_RIEMANNIAN else "Adam (fallback)"
        print(f"Training Poincaré embeddings on {self.device}...")
        print(f"Nodes: {num_nodes}, Edges: {len(edges)}, Dim: {config['embedding_dim']}")
        print(f"Optimizer: {optimizer_type}")
        
        self.model.train()
        
        for epoch in range(config['epochs']):
            total_loss = 0
            num_batches = 0
            
            if epoch < config['burn_in']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['burn_in_lr']
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['learning_rate']
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
            
            for batch in pbar:
                child_ids, parent_ids, negative_ids = batch
                
                child_ids = child_ids.to(self.device)
                parent_ids = parent_ids.to(self.device)
                negative_ids = negative_ids.to(self.device)
                
                optimizer.zero_grad()
                
                pos_distances = self.model(child_ids, parent_ids)
                
                batch_size = child_ids.size(0)
                child_ids_expanded = child_ids.unsqueeze(1).expand(-1, negative_ids.size(1))
                child_ids_flat = child_ids_expanded.reshape(-1)
                negative_ids_flat = negative_ids.reshape(-1)
                
                neg_distances = self.model(child_ids_flat, negative_ids_flat)
                neg_distances = neg_distances.reshape(batch_size, -1)
                
                loss = torch.mean(
                    torch.log(1 + torch.exp(-neg_distances + pos_distances.unsqueeze(1)))
                )
                
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    self.model.embeddings.weight.data = project_to_poincare_ball(
                        self.model.embeddings.weight.data
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.4f}")
        
        print("Training complete!")
        
        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model.embeddings.weight.cpu().numpy()
        
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def _collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: list of (child_id, parent_id, negatives)
        
        Returns:
            tuple of tensors
        """
        child_ids = []
        parent_ids = []
        negative_ids = []
        
        for child_id, parent_id, negatives in batch:
            child_ids.append(child_id)
            parent_ids.append(parent_id)
            negative_ids.append(negatives)
        
        return (
            torch.LongTensor(child_ids),
            torch.LongTensor(parent_ids),
            torch.LongTensor(negative_ids)
        )
    
    def get_embedding(self, node):
        """
        Get embedding for a single node.
        
        Args:
            node: node name or ID
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if isinstance(node, str):
            if node not in self.node2id:
                return np.zeros(self.embedding_dim)
            node_id = self.node2id[node]
        else:
            node_id = node
        
        return self.embeddings[node_id]
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: path to save model
        """
        data = {
            'embeddings': self.embeddings,
            'node2id': self.node2id,
            'id2node': self.id2node,
            'embedding_dim': self.embedding_dim,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: path to load model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.node2id = data['node2id']
        self.id2node = data['id2node']
        self.embedding_dim = data['embedding_dim']
        self.config = data['config']
        
        print(f"Model loaded from {filepath}")
        print(f"Embeddings shape: {self.embeddings.shape}")
