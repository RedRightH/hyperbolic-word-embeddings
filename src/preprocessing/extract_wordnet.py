import nltk
from nltk.corpus import wordnet as wn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR

def extract_hypernym_pairs(pos_filter=None, max_depth=None, limit=None):
    """
    Extract hypernym (is-a) relationships from WordNet.
    
    Args:
        pos_filter: Part of speech filter ('n', 'v', 'a', 'r') or None for all
        max_depth: Maximum depth in hierarchy to explore
        limit: Maximum number of pairs to extract
    
    Returns:
        list of (child, parent) tuples
    """
    print("Extracting hypernym pairs from WordNet...")
    
    pairs = []
    synsets = list(wn.all_synsets(pos=pos_filter))
    
    for synset in synsets:
        if limit and len(pairs) >= limit:
            break
            
        if max_depth:
            depth = synset.min_depth()
            if depth > max_depth:
                continue
        
        hypernyms = synset.hypernyms()
        
        for hypernym in hypernyms:
            child_name = synset.name()
            parent_name = hypernym.name()
            pairs.append((child_name, parent_name))
    
    print(f"Extracted {len(pairs)} hypernym pairs")
    return pairs

def save_edges(pairs, filename="wordnet_edges.txt"):
    """
    Save edge pairs to a text file.
    
    Args:
        pairs: list of (child, parent) tuples
        filename: output filename
    """
    output_path = RAW_DATA_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for child, parent in pairs:
            f.write(f"{child}\t{parent}\n")
    
    print(f"Saved {len(pairs)} edges to {output_path}")
    return output_path

def extract_and_save(pos_filter='n', max_depth=10, limit=10000, filename="wordnet_edges.txt"):
    """
    Extract hypernym pairs and save to file.
    
    Args:
        pos_filter: Part of speech filter
        max_depth: Maximum depth in hierarchy
        limit: Maximum number of pairs
        filename: output filename
    
    Returns:
        path to saved file
    """
    pairs = extract_hypernym_pairs(pos_filter=pos_filter, max_depth=max_depth, limit=limit)
    
    unique_pairs = list(set(pairs))
    print(f"Unique pairs: {len(unique_pairs)}")
    
    output_path = save_edges(unique_pairs, filename)
    
    unique_nodes = set()
    for child, parent in unique_pairs:
        unique_nodes.add(child)
        unique_nodes.add(parent)
    print(f"Unique nodes: {len(unique_nodes)}")
    
    return output_path

if __name__ == "__main__":
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("WordNet not found. Downloading...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    extract_and_save(pos_filter='n', max_depth=10, limit=10000)
