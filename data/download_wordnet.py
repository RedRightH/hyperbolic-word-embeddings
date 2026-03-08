import nltk
import os
from pathlib import Path

def download_wordnet():
    """
    Download WordNet data using NLTK.
    """
    print("Downloading WordNet data...")
    try:
        nltk.download('wordnet', quiet=False)
        nltk.download('omw-1.4', quiet=False)
        print("WordNet data downloaded successfully!")
        
        from nltk.corpus import wordnet as wn
        print(f"\nWordNet statistics:")
        print(f"Total synsets: {len(list(wn.all_synsets()))}")
        print(f"Total nouns: {len(list(wn.all_synsets('n')))}")
        print(f"Total verbs: {len(list(wn.all_synsets('v')))}")
        
    except Exception as e:
        print(f"Error downloading WordNet: {e}")
        raise

if __name__ == "__main__":
    download_wordnet()
