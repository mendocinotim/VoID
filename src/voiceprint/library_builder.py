import json
import os
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf
import logging
from pathlib import Path

from src.audio.audio_loader import AudioLoader
from src.audio.audio_preprocessing import AudioPreprocessor
from src.voiceprint.extractor import VoiceprintExtractor

logger = logging.getLogger(__name__)

class VoiceprintLibraryBuilder:
    """Builds a voiceprint library from whisper files and corresponding wav files."""
    
    def __init__(self, config):
        """Initialize the library builder."""
        self.config = config
        self.audio_loader = AudioLoader()
        self.preprocessor = AudioPreprocessor()
        self.extractor = VoiceprintExtractor(
            embedding_size=self.config.get('EMBEDDING_SIZE')
        )
    
    def parse_time(self, timestr):
        # Format: "HH:MM:SS,mmm"
        h, m, s_ms = timestr.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_segments_from_whisper(self, whisper_file: str) -> List[Dict]:
        """
        Extract voice segments from a whisper file.
        
        Args:
            whisper_file: Path to the whisper JSON file
            
        Returns:
            List of dictionaries containing segment information:
            {
                'speaker': str,
                'start': float,
                'end': float,
                'text': str
            }
        """
        try:
            with open(whisper_file, 'r', encoding='utf-8') as f:
                whisper_data = json.load(f)
            
            segments = []
            for segment in whisper_data.get('lines', []):
                if 'speakerDesignation' in segment:
                    segments.append({
                        'speaker': segment['speakerDesignation'],
                        'start': self.parse_time(segment['startTime']),
                        'end': self.parse_time(segment['endTime']),
                        'text': segment.get('text', '')
                    })
            
            return segments
        except Exception as e:
            logger.error(f"Error extracting segments from whisper file: {str(e)}")
            raise
    
    def extract_voiceprint(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract voiceprint from audio segment.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Voiceprint embedding
        """
        features = self.preprocessor.extract_features(audio, sr)
        return self.extractor.extract_embedding(features)
    
    def build_library(self, whisper_file: str, wav_file: str, output_dir: str) -> Dict[str, List[np.ndarray]]:
        """
        Build voiceprint library from whisper file and corresponding wav file.
        
        Args:
            whisper_file: Path to whisper JSON file
            wav_file: Path to corresponding WAV file
            output_dir: Directory to save voiceprint data
            
        Returns:
            Dictionary mapping speaker names to lists of voiceprint embeddings
        """
        try:
            # Extract segments from whisper file
            segments = self.extract_segments_from_whisper(whisper_file)
            if not segments:
                logger.warning(f"No speaker segments found in {whisper_file}")
                return {}
            
            # Load audio file
            audio, sr = self.audio_loader.load_audio(
                wav_file,
                target_sr=self.config.get('SAMPLE_RATE')
            )
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker = segment['speaker']
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
            
            # Extract voiceprints for each speaker
            voiceprint_library = {}
            for speaker, speaker_segs in speaker_segments.items():
                voiceprints = []
                
                for segment in speaker_segs:
                    # Extract audio segment
                    start_sample = int(segment['start'] * sr)
                    end_sample = int(segment['end'] * sr)
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Extract voiceprint
                    voiceprint = self.extract_voiceprint(segment_audio, sr)
                    voiceprints.append(voiceprint)
                
                # Save voiceprints for this speaker
                speaker_dir = os.path.join(output_dir, speaker)
                os.makedirs(speaker_dir, exist_ok=True)
                
                for i, voiceprint in enumerate(voiceprints):
                    output_file = os.path.join(speaker_dir, f"voiceprint_{i}.npy")
                    np.save(output_file, voiceprint)
                
                voiceprint_library[speaker] = voiceprints
            
            return voiceprint_library
            
        except Exception as e:
            logger.error(f"Error building voiceprint library: {str(e)}")
            raise
    
    def batch_build_library(self, whisper_dir: str, wav_dir: str, output_dir: str) -> Dict[str, List[np.ndarray]]:
        """
        Build voiceprint library from multiple whisper and wav files.
        
        Args:
            whisper_dir: Directory containing whisper JSON files
            wav_dir: Directory containing corresponding WAV files
            output_dir: Directory to save voiceprint data
            
        Returns:
            Dictionary mapping speaker names to lists of voiceprint embeddings
        """
        voiceprint_library = {}
        
        for whisper_file in os.listdir(whisper_dir):
            if not whisper_file.endswith('.json'):
                continue
            
            # Find corresponding wav file
            base_name = os.path.splitext(whisper_file)[0]
            wav_file = os.path.join(wav_dir, f"{base_name}.wav")
            
            if not os.path.exists(wav_file):
                logger.warning(f"No corresponding WAV file found for {whisper_file}")
                continue
            
            try:
                # Build library for this file pair
                file_library = self.build_library(
                    os.path.join(whisper_dir, whisper_file),
                    wav_file,
                    output_dir
                )
                
                # Merge with main library
                for speaker, voiceprints in file_library.items():
                    if speaker not in voiceprint_library:
                        voiceprint_library[speaker] = []
                    voiceprint_library[speaker].extend(voiceprints)
                
            except Exception as e:
                logger.error(f"Error processing {whisper_file}: {str(e)}")
                continue
        
        return voiceprint_library 