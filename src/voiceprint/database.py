import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import pickle

class VoiceprintDatabase:
    """Manages storage and retrieval of speaker voiceprints."""
    
    def __init__(self, db_path: str):
        """
        Initialize the voiceprint database.
        
        Args:
            db_path: Path to the database directory
        """
        self.db_path = db_path
        self.embeddings_path = os.path.join(db_path, 'embeddings')
        self.metadata_path = os.path.join(db_path, 'metadata.json')
        
        # Create database directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Load or create metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()
    
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
        # Generate unique ID for the embedding
        embedding_id = f"{speaker_name}_{len(self.metadata.get(speaker_name, []))}"
        
        # Save embedding
        embedding_path = os.path.join(self.embeddings_path, f"{embedding_id}.npy")
        np.save(embedding_path, embedding)
        
        # Update metadata
        if speaker_name not in self.metadata:
            self.metadata[speaker_name] = []
            
        self.metadata[speaker_name].append({
            'id': embedding_id,
            'path': embedding_path,
            'metadata': metadata or {}
        })
        
        self._save_metadata()
    
    def get_voiceprints(self, speaker_name: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
        """
        Get voiceprints from the database.
        
        Args:
            speaker_name: Optional speaker name to filter by
            
        Returns:
            Dictionary mapping speaker names to lists of embeddings
        """
        result = {}
        
        if speaker_name:
            if speaker_name in self.metadata:
                embeddings = []
                for entry in self.metadata[speaker_name]:
                    embedding = np.load(entry['path'])
                    embeddings.append(embedding)
                result[speaker_name] = embeddings
        else:
            for name, entries in self.metadata.items():
                embeddings = []
                for entry in entries:
                    embedding = np.load(entry['path'])
                    embeddings.append(embedding)
                result[name] = embeddings
                
        return result
    
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
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2) 