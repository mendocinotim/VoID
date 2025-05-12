import numpy as np
from typing import Dict, List, Optional, Tuple
from .extractor import VoiceprintExtractor

class VoiceprintMatcher:
    """Handles matching of speaker voiceprints for identification."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 extractor: Optional[VoiceprintExtractor] = None):
        """
        Initialize the voiceprint matcher.
        
        Args:
            similarity_threshold: Threshold for considering a match
            extractor: Optional voiceprint extractor instance
        """
        self.similarity_threshold = similarity_threshold
        self.extractor = extractor or VoiceprintExtractor()
    
    def find_best_match(self,
                       query_embedding: np.ndarray,
                       database: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float]:
        """
        Find the best matching speaker for a query embedding.
        
        Args:
            query_embedding: Query speaker embedding
            database: Dictionary mapping speaker names to lists of embeddings
            
        Returns:
            Tuple of (best matching speaker name, similarity score)
            Returns (None, 0.0) if no match above threshold
        """
        best_score = 0.0
        best_speaker = None
        
        for speaker_name, embeddings in database.items():
            # Compute average similarity across all embeddings for this speaker
            scores = []
            for embedding in embeddings:
                score = self.extractor.compute_similarity(query_embedding, embedding)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_speaker = speaker_name
        
        if best_score >= self.similarity_threshold:
            return best_speaker, best_score
        else:
            return None, 0.0
    
    def identify_speakers(self,
                         embeddings: List[np.ndarray],
                         database: Dict[str, List[np.ndarray]]) -> List[Tuple[Optional[str], float]]:
        """
        Identify speakers for a list of embeddings.
        
        Args:
            embeddings: List of speaker embeddings to identify
            database: Dictionary mapping speaker names to lists of embeddings
            
        Returns:
            List of tuples (speaker_name, similarity_score) for each embedding
        """
        results = []
        for embedding in embeddings:
            speaker, score = self.find_best_match(embedding, database)
            results.append((speaker, score))
        return results
    
    def compute_similarity_matrix(self,
                                embeddings1: List[np.ndarray],
                                embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First list of embeddings
            embeddings2: Second list of embeddings
            
        Returns:
            Matrix of similarity scores
        """
        n1 = len(embeddings1)
        n2 = len(embeddings2)
        similarity_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                similarity_matrix[i, j] = self.extractor.compute_similarity(
                    embeddings1[i], embeddings2[j]
                )
                
        return similarity_matrix
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the similarity threshold for matching.
        
        Args:
            threshold: New similarity threshold
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        self.similarity_threshold = threshold 