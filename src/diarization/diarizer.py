import numpy as np
from typing import List, Dict, Optional
from pyannote.audio import Pipeline
import torch
import os

class Diarizer:
    """Handles speaker diarization using pyannote.audio."""
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the diarizer.
        
        Args:
            auth_token: HuggingFace authentication token for pyannote.audio
        """
        self.auth_token = auth_token or os.getenv("HF_AUTH_TOKEN")
        if not self.auth_token:
            raise ValueError("HuggingFace authentication token is required")
            
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=self.auth_token
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to(torch.device("cuda"))
    
    def diarize(self, 
                audio: np.ndarray,
                sample_rate: int,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> List[Dict]:
        """
        Perform speaker diarization on the audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            List of dictionaries containing segment information:
            [{'start': float, 'end': float, 'speaker_id': str}, ...]
        """
        # Convert numpy array to torch tensor and reshape to (channel, time)
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        
        # Prepare diarization parameters
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
            
        # Perform diarization
        diarization = self.pipeline(
            {"waveform": audio_tensor, "sample_rate": sample_rate},
            **diarization_params
        )
        
        # Convert diarization results to list of dictionaries
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker_id': speaker
            })
            
        return segments
    
    def get_speaker_segments(self, 
                           audio: np.ndarray,
                           sample_rate: int,
                           speaker_id: str) -> List[Dict]:
        """
        Get all segments for a specific speaker.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            speaker_id: ID of the speaker to find segments for
            
        Returns:
            List of dictionaries containing segment information for the specified speaker
        """
        segments = self.diarize(audio, sample_rate)
        return [seg for seg in segments if seg['speaker_id'] == speaker_id] 