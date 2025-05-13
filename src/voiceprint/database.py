import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import pickle

class VoiceprintDatabase:
    """Manages storage and retrieval of speaker voiceprints."""
    
    def __init__(self, db_path=None):
        """
        Initialize the voiceprint database.
        
        Args:
            db_path: Path to the database directory
        """
        self.db_path = db_path or os.path.expanduser("~/.voice_diarization_db")
        self.embeddings_dir = os.path.join(self.db_path, "embeddings")
        self.metadata_file = os.path.join(self.db_path, "metadata.json")
        self._ensure_dirs()
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_voiceprint(self, 
                      speaker_name: str,
                      embedding: np.ndarray,
                      metadata: Optional[Dict] = None) -> None:
        """
        Add a new voiceprint to the database.
        
        Args:
            speaker_name: Name of the speaker
            embedding: Speaker embedding
            metadata: Additional metadata about the voiceprint
        """
        if speaker_name not in self.metadata:
            self.metadata[speaker_name] = []
        
        # Generate unique filename for embedding
        embedding_id = len(self.metadata[speaker_name])
        embedding_path = os.path.join(self.embeddings_dir, f"{speaker_name}_{embedding_id}.npy")
        
        # Save embedding
        np.save(embedding_path, embedding)
        
        # Save metadata
        metadata_entry = {
            'path': embedding_path,
            'metadata': metadata or {},
            'ignored': False  # Default to not ignored
        }
        self.metadata[speaker_name].append(metadata_entry)
        self._save_metadata()
    
    def get_voiceprints(self, include_ignored=False) -> Dict[str, List[np.ndarray]]:
        """
        Get voiceprints from the database.
        
        Args:
            include_ignored: Whether to include ignored voiceprints
            
        Returns:
            Dictionary mapping speaker names to lists of embeddings
        """
        voiceprints = {}
        for speaker, entries in self.metadata.items():
            if speaker not in voiceprints:
                voiceprints[speaker] = []
            
            for entry in entries:
                # Skip ignored entries unless explicitly requested
                if not include_ignored and entry.get('ignored', False):
                    continue
                    
                if os.path.exists(entry['path']):
                    embedding = np.load(entry['path'])
                    voiceprints[speaker].append(embedding)
        
        return voiceprints
    
    def remove_voiceprint(self, speaker_name: str, embedding_id: str) -> None:
        """
        Remove a voiceprint from the database.
        
        Args:
            speaker_name: Name of the speaker
            embedding_id: ID of the embedding to remove
        """
        if speaker_name in self.metadata:
            # Find and remove the embedding
            for i, entry in enumerate(self.metadata[speaker_name]):
                if entry['id'] == embedding_id:
                    # Remove the file
                    os.remove(entry['path'])
                    # Remove from metadata
                    self.metadata[speaker_name].pop(i)
                    break
            
            # Remove speaker entry if no more voiceprints
            if not self.metadata[speaker_name]:
                del self.metadata[speaker_name]
                
            self._save_metadata()
    
    def get_speaker_names(self) -> List[str]:
        """
        Get list of all speaker names in the database.
        
        Returns:
            List of speaker names
        """
        return list(self.metadata.keys())
    
    def mark_as_ignored(self, speaker, indices):
        """Mark specific voiceprints as ignored."""
        if speaker in self.metadata:
            for idx in indices:
                if 0 <= idx < len(self.metadata[speaker]):
                    self.metadata[speaker][idx]['ignored'] = True
            self._save_metadata()
    
    def mark_as_unignored(self, speaker, indices):
        """Mark specific voiceprints as not ignored."""
        if speaker in self.metadata:
            for idx in indices:
                if 0 <= idx < len(self.metadata[speaker]):
                    self.metadata[speaker][idx]['ignored'] = False
            self._save_metadata()
    
    def _ensure_dirs(self):
        """Ensure database directories exist."""
        os.makedirs(self.embeddings_dir, exist_ok=True) 