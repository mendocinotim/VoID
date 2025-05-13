from src.voiceprint.database import VoiceprintDatabase
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd

def analyze_embeddings(embeddings):
    """Analyze a list of embeddings and return statistics."""
    if not embeddings:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'shape': None
        }
    
    # Convert to numpy array for analysis
    emb_array = np.array(embeddings)
    
    return {
        'count': len(embeddings),
        'mean': np.mean(emb_array),
        'std': np.std(emb_array),
        'min': np.min(emb_array),
        'max': np.max(emb_array),
        'shape': emb_array.shape
    }

def summarize_voiceprints(db_path):
    db = VoiceprintDatabase(db_path=db_path)
    voiceprints = db.get_voiceprints()
    summary = []
    for speaker, embeddings in voiceprints.items():
        files = set()
        for emb in embeddings:
            # Try to get file info if available (if emb is a dict with 'file' or 'path')
            if isinstance(emb, dict):
                if 'file' in emb:
                    files.add(emb['file'])
                elif 'path' in emb:
                    files.add(os.path.basename(emb['path']))
        summary.append({
            'Speaker': speaker,
            'NumVoiceprints': len(embeddings),
            'NumFiles': len(files) if files else 'N/A',
            'Files': ', '.join(sorted(files)) if files else 'N/A'
        })
    df = pd.DataFrame(summary)
    print("\n=== Voiceprint Database Summary ===")
    print(df.to_string(index=False))
    return df

def plot_embedding_distribution(embeddings, title):
    """Plot the distribution of embedding values."""
    if not embeddings:
        print(f"No embeddings to plot for {title}")
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(embeddings).flatten(), bins=50)
    plt.title(f'Embedding Distribution - {title}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'embedding_dist_{title.replace(" ", "_")}.png')
    plt.close()

def check_db_path(db_path):
    """Check if database exists and is accessible."""
    path = Path(db_path)
    if not path.exists():
        return False, f"Path does not exist: {db_path}"
    
    if not path.is_dir():
        return False, f"Path is not a directory: {db_path}"
    
    # Check for database files
    db_files = list(path.glob('*.json'))
    if not db_files:
        return False, f"No database files found in: {db_path}"
    
    return True, f"Database found at: {db_path} with {len(db_files)} files"

def list_blank_speaker_files(db_path):
    db = VoiceprintDatabase(db_path=db_path)
    voiceprints = db.get_voiceprints()
    blanks = voiceprints.get('', [])
    if not blanks:
        print("\nNo blank speaker entries found.")
        return
    print("\n=== Files for Blank Speaker Entries ===")
    for i, emb in enumerate(blanks):
        # Try to get metadata if available
        meta = None
        if hasattr(db, 'metadata') and '' in db.metadata:
            try:
                meta = db.metadata[''][i]
            except Exception:
                meta = None
        print(f"Entry {i+1}:")
        if meta:
            print(f"  Path: {meta.get('path', 'N/A')}")
            print(f"  Whisper file: {meta.get('metadata', {}).get('whisper_file', 'N/A')}")
            print(f"  WAV file: {meta.get('metadata', {}).get('wav_file', 'N/A')}")
            print(f"  Start: {meta.get('metadata', {}).get('start', 'N/A')}")
            print(f"  End: {meta.get('metadata', {}).get('end', 'N/A')}")
            print(f"  Text: {meta.get('metadata', {}).get('text', 'N/A')}")
        else:
            print("  (No metadata available)")
        print()

def main():
    DB_PATH = os.path.expanduser("~/.voice_diarization_db")
    summarize_voiceprints(DB_PATH)
    list_blank_speaker_files(DB_PATH)

if __name__ == "__main__":
    main() 