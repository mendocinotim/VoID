import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional

class AudioLoader:
    """Handles loading and basic validation of audio files."""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'}
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and convert it to mono if necessary.
        
        Args:
            file_path: Path to the audio file
            target_sr: Target sample rate (default: 16000 Hz)
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in AudioLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        return audio, sr
    
    @staticmethod
    def validate_audio(audio: np.ndarray, sample_rate: int) -> bool:
        """
        Validate audio data for processing.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            bool: True if audio is valid, False otherwise
        """
        if audio is None or len(audio) == 0:
            return False
            
        if not isinstance(audio, np.ndarray):
            return False
            
        if sample_rate <= 0:
            return False
            
        return True
    
    @staticmethod
    def save_audio(audio: np.ndarray, sample_rate: int, file_path: str) -> None:
        """
        Save audio data to a file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            file_path: Path to save the audio file
        """
        sf.write(file_path, audio, sample_rate) 