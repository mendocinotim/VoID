import json
import os
import numpy as np
from typing import Dict, List, Tuple
import soundfile as sf
import logging
from pathlib import Path
import pandas as pd
from collections import defaultdict

from src.audio.audio_loader import AudioLoader
from src.audio.audio_preprocessing import AudioPreprocessor
from src.voiceprint.extractor import VoiceprintExtractor
from src.voiceprint.database import VoiceprintDatabase

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
        # Initialize the central database
        self.db = VoiceprintDatabase(db_path=self.config.get('DB_PATH'))
    
    def parse_time(self, timestr):
        # Format: "HH:MM:SS,mmm"
        h, m, s_ms = timestr.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_segments_from_js_transcript(self, js_file: str) -> List[Dict]:
        """
        Extract speaker segments from a .js (JSON array) transcript file.
        """
        try:
            with open(js_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = []
            for entry in data:
                segments.append({
                    'speaker': entry.get('speaker', 'Unknown'),
                    'start': None,
                    'end': None,
                    'text': entry.get('text', '')
                })
            return segments
        except Exception as e:
            logger.error(f"Error extracting segments from js transcript: {str(e)}")
            raise

    def extract_segments_from_whisper(self, whisper_file: str) -> List[Dict]:
        """
        Extract voice segments from a whisper file or .js transcript.
        Args:
            whisper_file: Path to the whisper JSON or JS file
        Returns:
            List of dictionaries containing segment information:
            {
                'speaker': str,
                'start': float or None,
                'end': float or None,
                'text': str
            }
        """
        try:
            # Detect file type by extension or structure
            if whisper_file.endswith('.js'):
                return self.extract_segments_from_js_transcript(whisper_file)
            with open(whisper_file, 'r', encoding='utf-8') as f:
                whisper_data = json.load(f)
            # If it's a list, treat as .js format
            if isinstance(whisper_data, list):
                segments = []
                for entry in whisper_data:
                    segments.append({
                        'speaker': entry.get('speaker', 'Unknown'),
                        'start': None,
                        'end': None,
                        'text': entry.get('text', '')
                    })
                return segments
            # Otherwise, treat as MacWhisper JSON with 'lines'
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
                    # Skip segments with missing start/end
                    if segment['start'] is None or segment['end'] is None:
                        logger.warning(f"Skipping segment with missing start/end in {whisper_file}: {segment}")
                        continue
                    # Extract audio segment
                    start_sample = int(segment['start'] * sr)
                    end_sample = int(segment['end'] * sr)
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Extract voiceprint
                    voiceprint = self.extract_voiceprint(segment_audio, sr)
                    voiceprints.append(voiceprint)
                    # Save to central database
                    self.db.add_voiceprint(
                        speaker_name=speaker,
                        embedding=voiceprint,
                        metadata={
                            'whisper_file': whisper_file,
                            'wav_file': wav_file,
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': segment.get('text', '')
                        }
                    )
                
                # Save voiceprints for this speaker (optional: local dir)
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

    def batch_process_range(self, root_dir: str, start_ref: str, end_ref: str):
        """
        Batch process a range of reference directories.
        Returns summary tables as described by the user.
        """
        # Get sorted list of directories
        all_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        # Find start and end indices
        try:
            start_idx = all_dirs.index(start_ref)
            end_idx = all_dirs.index(end_ref)
        except ValueError as e:
            raise Exception(f"Start or end reference not found: {e}")
        selected_dirs = all_dirs[start_idx:end_idx+1]

        summary_rows = []  # For first table
        speaker_sample_counts = defaultdict(int)
        speaker_file_counts = defaultdict(set)

        for ref_dir in selected_dirs:
            ref_path = os.path.join(root_dir, ref_dir)
            audio_dir = os.path.join(ref_path, 'AUDIO')
            json_dir = os.path.join(ref_path, 'JSON')
            # Find .wav file
            wav_file = None
            for f in os.listdir(audio_dir):
                if f.lower().endswith('.wav'):
                    wav_file = os.path.join(audio_dir, f)
                    break
            # Find transcript file (.json, .js, .json.js)
            transcript_file = None
            for ext in ['.json', '.js', '.json.js']:
                for f in os.listdir(json_dir):
                    if f.lower().endswith(ext):
                        transcript_file = os.path.join(json_dir, f)
                        break
                if transcript_file:
                    break
            if not wav_file or not transcript_file:
                continue  # Skip if missing
            # Call build_library to extract and save voiceprints
            self.build_library(transcript_file, wav_file, ref_path)
            # Extract segments for summary
            segments = self.extract_segments_from_whisper(transcript_file)
            for seg in segments:
                speaker = seg['speaker']
                summary_rows.append({'RefDir': ref_dir, 'SpeakerName': speaker})
                speaker_sample_counts[speaker] += 1
                speaker_file_counts[speaker].add(ref_dir)
        # First table: one row per reference dir processed
        df1 = pd.DataFrame(summary_rows)
        df1 = df1.sort_values(['RefDir', 'SpeakerName']).reset_index(drop=True)
        # Second table: one row per speaker
        df2 = pd.DataFrame([
            {
                'SpeakerName': speaker,
                'TotalSamples': speaker_sample_counts[speaker],
                'TotalFiles': len(speaker_file_counts[speaker])
            }
            for speaker in speaker_sample_counts
        ])
        df2 = df2.sort_values(['SpeakerName']).reset_index(drop=True)
        return df1, df2 