import os
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from src.voiceprint.library_builder import VoiceprintLibraryBuilder
from src.voiceprint.database import VoiceprintDatabase
from src.audio.audio_preprocessing import AudioPreprocessor
from src.voiceprint.extractor import VoiceprintExtractor
from src.config.settings import Config
import time
from tqdm import tqdm
import pandas as pd

class InterviewProcessor:
    def __init__(self):
        self.config = Config()
        self.db = VoiceprintDatabase(db_path=os.path.expanduser("~/.voice_diarization_db"))
        self.library_builder = VoiceprintLibraryBuilder(config=self.config)
        
    def get_interview_dirs(self, root_dir, start_num=1, end_num=42):
        """Get list of interview directories in the specified range."""
        dirs = []
        for item in os.listdir(root_dir):
            if item.startswith('.') or not os.path.isdir(os.path.join(root_dir, item)):
                continue
            try:
                num = int(item.split('_')[0])
                if start_num <= num <= end_num:
                    dirs.append((num, item))
            except (ValueError, IndexError):
                continue
        return sorted(dirs, key=lambda x: x[0])
    
    def process_interview(self, interview_dir):
        """Process a single interview directory."""
        print(f"\nProcessing interview: {interview_dir}")
        
        # Check for required subdirectories
        audio_dir = os.path.join(interview_dir, "AUDIO")
        json_dir = os.path.join(interview_dir, "JSON")
        
        if not os.path.exists(audio_dir) or not os.path.exists(json_dir):
            print(f"Missing required directories in {interview_dir}")
            return False
        
        # Find transcript file
        transcript_files = [f for f in os.listdir(json_dir) if f.endswith(('.json', '.js'))]
        if not transcript_files:
            print(f"No transcript files found in {json_dir}")
            return False
        
        transcript_file = os.path.join(json_dir, transcript_files[0])
        
        # Find audio file
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return False
        
        audio_file = os.path.join(audio_dir, audio_files[0])
        
        try:
            # Extract speaker segments and build voiceprints
            success = self.library_builder.process_file(
                audio_path=audio_file,
                transcript_path=transcript_file,
                reference_dir=interview_dir
            )
            
            if success:
                print(f"Successfully processed {interview_dir}")
                return True
            else:
                print(f"Failed to process {interview_dir}")
                return False
                
        except Exception as e:
            print(f"Error processing {interview_dir}: {str(e)}")
            return False

def analyze_problematic_entries(df1, df2):
    """Analyze entries that need attention and show their source directories."""
    print("\n=== Analyzing Entries Needing Attention ===")
    
    # Get the problematic indices from df2
    problematic_speakers = df2.iloc[[6, 13, 14, 16]]['SpeakerName'].tolist()
    print("\nProblematic Speakers:")
    for speaker in problematic_speakers:
        print(f"- {speaker}")
    
    # Find all entries in df1 for these speakers
    problematic_entries = df1[df1['SpeakerName'].isin(problematic_speakers)]
    
    # Group by RefDir and SpeakerName to show unique combinations
    grouped = problematic_entries.groupby(['RefDir', 'SpeakerName']).size().reset_index(name='count')
    
    print("\nDirectories Needing Attention:")
    for _, row in grouped.iterrows():
        print(f"\nDirectory: {row['RefDir']}")
        print(f"Speaker: {row['SpeakerName']}")
        print(f"Number of segments: {row['count']}")
        
        # Construct the full path to help locate the files
        full_path = os.path.join("/Volumes/AI_ETS_2TB/RawInterviews_forTranscription/Transcribed Raw Interviews/transcriptions", 
                                row['RefDir'])
        print(f"Full path: {full_path}")
        
        # List the contents of the AUDIO and JSON directories
        audio_dir = os.path.join(full_path, "AUDIO")
        json_dir = os.path.join(full_path, "JSON")
        
        if os.path.exists(audio_dir):
            print("\nAudio files:")
            for f in os.listdir(audio_dir):
                if f.endswith('.wav'):
                    print(f"  - {f}")
        
        if os.path.exists(json_dir):
            print("\nTranscript files:")
            for f in os.listdir(json_dir):
                if f.endswith(('.json', '.js')):
                    print(f"  - {f}")

def main():
    # Root directory containing all interviews
    ROOT_DIR = "/Volumes/AI_ETS_2TB/RawInterviews_forTranscription/Transcribed Raw Interviews/transcriptions"
    START_REF = "1_Johnny Mandel"
    END_REF = "42_Steve Allen"

    # Initialize config and builder
    config = Config()
    builder = VoiceprintLibraryBuilder(config)

    print(f"Processing from {START_REF} to {END_REF} in {ROOT_DIR}...")
    df1, df2 = builder.batch_process_range(ROOT_DIR, START_REF, END_REF)

    # Print results
    print("\nTable 1: RefDir/SpeakerName")
    print(df1)
    print("\nTable 2: Speaker Summary")
    print(df2)
    
    # Analyze problematic entries
    analyze_problematic_entries(df1, df2)

if __name__ == "__main__":
    main() 