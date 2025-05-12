import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import os

class SpeakerEmbeddingModel(nn.Module):
    """Simple CNN-based speaker embedding model."""
    
    def __init__(self, input_size: int = 40, embedding_size: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, embedding_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class VoiceprintExtractor:
    """Handles extraction of speaker embeddings (voiceprints)."""
    
    def __init__(self, 
                 embedding_size: int = 128,
                 device: Optional[str] = None):
        """
        Initialize the voiceprint extractor.
        
        Args:
            embedding_size: Size of the speaker embedding
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpeakerEmbeddingModel(embedding_size=embedding_size)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def extract_embedding(self, 
                         features: np.ndarray,
                         normalize: bool = True) -> np.ndarray:
        """
        Extract speaker embedding from audio features.
        
        Args:
            features: Input features (e.g., MFCCs)
            normalize: Whether to normalize the embedding
            
        Returns:
            Speaker embedding as numpy array
        """
        with torch.no_grad():
            # Convert to torch tensor and add batch dimension
            features_tensor = torch.from_numpy(features).float()
            if len(features_tensor.shape) == 2:
                features_tensor = features_tensor.unsqueeze(0)
            features_tensor = features_tensor.to(self.device)
            
            # Extract embedding
            embedding = self.model(features_tensor)
            
            # Convert to numpy and normalize if requested
            embedding = embedding.cpu().numpy()
            if normalize:
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                
            return embedding.squeeze()
    
    def compute_similarity(self, 
                          embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def save_model(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device)) 