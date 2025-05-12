import numpy as np
from typing import Tuple
import librosa

class AudioPreprocessor:
    """Handles audio preprocessing tasks."""
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have zero mean and unit variance.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        return librosa.util.normalize(audio)
    
    @staticmethod
    def remove_silence(audio: np.ndarray, 
                      top_db: float = 20,
                      frame_length: int = 2048,
                      hop_length: int = 512) -> np.ndarray:
        """
        Remove silent parts from audio.
        
        Args:
            audio: Input audio array
            top_db: Threshold in decibels below reference to consider as silence
            frame_length: Length of each frame for STFT
            hop_length: Number of samples between successive frames
            
        Returns:
            Audio array with silence removed
        """
        return librosa.effects.trim(audio, top_db=top_db, 
                                  frame_length=frame_length,
                                  hop_length=hop_length)[0]
    
    @staticmethod
    def preprocess_audio(audio: np.ndarray, 
                        sample_rate: int,
                        normalize: bool = True,
                        remove_silence: bool = True) -> np.ndarray:
        """
        Apply a series of preprocessing steps to the audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            normalize: Whether to normalize the audio
            remove_silence: Whether to remove silence
            
        Returns:
            Preprocessed audio array
        """
        processed_audio = audio.copy()
        
        if remove_silence:
            processed_audio = AudioPreprocessor.remove_silence(processed_audio)
            
        if normalize:
            processed_audio = AudioPreprocessor.normalize_audio(processed_audio)
            
        return processed_audio
    
    @staticmethod
    def extract_features(audio: np.ndarray, 
                        sample_rate: int,
                        n_mfcc: int = 13,
                        n_fft: int = 2048,
                        hop_length: int = 512) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            
        Returns:
            MFCC features array
        """
        mfccs = librosa.feature.mfcc(y=audio, 
                                    sr=sample_rate,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        return mfccs 