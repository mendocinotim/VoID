import streamlit as st
import numpy as np
import os
import tempfile
from typing import List, Dict, Optional
import librosa
import soundfile as sf
import logging
import traceback
import time
from pathlib import Path
import matplotlib.pyplot as plt

from src.audio.audio_loader import AudioLoader
from src.audio.audio_preprocessing import AudioPreprocessor
from src.diarization.diarizer import Diarizer
from src.voiceprint.extractor import VoiceprintExtractor
from src.voiceprint.database import VoiceprintDatabase
from src.voiceprint.matcher import VoiceprintMatcher
from src.voiceprint.library_builder import VoiceprintLibraryBuilder
from src.config.settings import Config

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_message(msg):
    if 'log' not in st.session_state:
        st.session_state['log'] = []
    st.session_state['log'].append(msg)
    logger.info(msg)

def process_large_audio(audio_path: str, chunk_size: int = 300):
    """Process large audio files in chunks to avoid memory issues."""
    try:
        # Get total duration
        info = sf.info(audio_path)
        total_duration = info.duration
        sample_rate = info.samplerate
        
        # Process in chunks
        chunks = []
        for start_time in range(0, int(total_duration), chunk_size):
            end_time = min(start_time + chunk_size, total_duration)
            
            # Load chunk
            audio_chunk, _ = librosa.load(
                audio_path,
                sr=sample_rate,
                offset=start_time,
                duration=end_time - start_time
            )
            
            chunks.append(audio_chunk)
            
            # Update progress
            progress = min(1.0, (end_time / total_duration))
            yield progress, f"Processing chunk {start_time}-{end_time}s..."
        
        # Combine chunks
        audio = np.concatenate(chunks)
        yield 1.0, (audio, sample_rate)
        
    except Exception as e:
        logger.error(f"Error processing large audio: {str(e)}")
        raise

class VoiceDiarizationApp:
    """Streamlit-based UI for voice diarization and identification."""
    
    def __init__(self):
        """Initialize the application."""
        try:
            log_message("Initializing application...")
            self.config = Config()
            
            # Create database directory if it doesn't exist
            db_path = self.config.get('DB_PATH')
            os.makedirs(db_path, exist_ok=True)
            log_message(f"Database path: {db_path}")
            
            # Get Hugging Face token from Streamlit secrets
            hf_token = st.secrets.get("HF_AUTH_TOKEN")
            if not hf_token:
                st.error("Hugging Face token not found in secrets. Please add it to .streamlit/secrets.toml")
                st.stop()
            log_message("Hugging Face token loaded")
            
            self.audio_loader = AudioLoader()
            self.preprocessor = AudioPreprocessor()
            self.diarizer = Diarizer(auth_token=hf_token)
            self.extractor = VoiceprintExtractor(
                embedding_size=self.config.get('EMBEDDING_SIZE')
            )
            self.database = VoiceprintDatabase(
                db_path=db_path
            )
            self.matcher = VoiceprintMatcher(
                similarity_threshold=self.config.get('SIMILARITY_THRESHOLD'),
                extractor=self.extractor
            )
            self.library_builder = VoiceprintLibraryBuilder(self.config)
            log_message("All components initialized successfully")
            
            # Initialize session state
            if 'audio_data' not in st.session_state:
                st.session_state.audio_data = None
            if 'sample_rate' not in st.session_state:
                st.session_state.sample_rate = None
            if 'segments' not in st.session_state:
                st.session_state.segments = None
            if 'log' not in st.session_state:
                st.session_state['log'] = []
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            
            # Add chunk size to config
            self.chunk_size = self.config.get('CHUNK_SIZE', 300)  # 5 minutes per chunk
            
        except Exception as e:
            log_message(f"Error during initialization: {str(e)}")
            log_message(traceback.format_exc())
            st.error(f"Error during initialization: {str(e)}")
            st.stop()
    
    def display_voiceprint_info(self, speaker_dir: str):
        """Display detailed information about a speaker's voiceprints."""
        voiceprint_dir = os.path.join(self.config.get('DB_PATH'), 'voiceprints', speaker_dir)
        if not os.path.exists(voiceprint_dir):
            return
        
        st.markdown(f"### Voiceprint Details for {speaker_dir}")
        
        # Get all voiceprint files
        voiceprints = [f for f in os.listdir(voiceprint_dir) if f.endswith('.npy')]
        if not voiceprints:
            st.markdown("No voiceprints found")
            return
        
        # Create a table of voiceprint information
        st.markdown("#### Voiceprint Files")
        for vp in voiceprints:
            vp_path = os.path.join(voiceprint_dir, vp)
            file_size = os.path.getsize(vp_path) / 1024  # Size in KB
            created_time = os.path.getctime(vp_path)
            created_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_time))
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{vp}**")
            with col2:
                st.markdown(f"{file_size:.1f} KB")
            with col3:
                st.markdown(created_date)
            
            # Add a button to view the voiceprint data
            if st.button(f"View Voiceprint Data", key=f"view_{vp}"):
                try:
                    voiceprint_data = np.load(vp_path)
                    st.markdown("#### Voiceprint Data")
                    st.write(f"Shape: {voiceprint_data.shape}")
                    st.write(f"Mean: {np.mean(voiceprint_data):.4f}")
                    st.write(f"Std: {np.std(voiceprint_data):.4f}")
                    
                    # Plot the voiceprint data
                    st.markdown("#### Voiceprint Visualization")
                    fig = plt.figure(figsize=(10, 4))
                    plt.plot(voiceprint_data)
                    plt.title(f"Voiceprint: {vp}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error loading voiceprint data: {str(e)}")

    def run(self):
        """Run the Streamlit application."""
        try:
            log_message("App started or rerun.")
            st.title("Voice Diarization and Identification")
            
            # Sidebar: Stored Voiceprints
            st.sidebar.subheader("Stored Voiceprints")
            voiceprint_dir = os.path.join(self.config.get('DB_PATH'), 'voiceprints')
            if os.path.exists(voiceprint_dir):
                for speaker_dir in os.listdir(voiceprint_dir):
                    speaker_path = os.path.join(voiceprint_dir, speaker_dir)
                    if os.path.isdir(speaker_path):
                        st.sidebar.markdown(f"### {speaker_dir}")
                        voiceprints = os.listdir(speaker_path)
                        if voiceprints:
                            for vp in voiceprints:
                                if vp.endswith('.npy'):
                                    st.sidebar.markdown(f"- {vp}")
                        else:
                            st.sidebar.markdown("No voiceprints found")
            else:
                st.sidebar.markdown("No voiceprints directory found")
            
            # Main area: Voiceprint Details
            st.subheader("Voiceprint Details")
            if os.path.exists(voiceprint_dir):
                speaker_dirs = [d for d in os.listdir(voiceprint_dir) 
                              if os.path.isdir(os.path.join(voiceprint_dir, d))]
                if speaker_dirs:
                    selected_speaker = st.selectbox(
                        "Select Speaker",
                        speaker_dirs,
                        format_func=lambda x: x
                    )
                    if selected_speaker:
                        self.display_voiceprint_info(selected_speaker)
                else:
                    st.markdown("No speakers found in voiceprint database")
            else:
                st.markdown("No voiceprint database found")
            
            # Sidebar: Execution/Error Log
            with st.sidebar.expander("Execution/Error Log", expanded=True):
                st.text_area("Log", value="\n".join(st.session_state['log']), height=200, key="log_display", disabled=True)
                if st.button("Clear Log"):
                    st.session_state['log'] = []
            
            # Sidebar: Start Again Button
            if st.sidebar.button("Start Again"):
                for key in ['audio_data', 'sample_rate', 'segments', 'log', 'processing']:
                    if key in st.session_state:
                        del st.session_state[key]
            
            # File upload section
            st.subheader("Upload Files")
            st.markdown("""
            Please upload both:
            1. The MacWhisper JSON file (contains speaker labels and timestamps)
            2. The corresponding WAV file (converted from AIFF)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                whisper_file = st.file_uploader(
                    "Upload MacWhisper JSON file",
                    type=['json'],
                    help="The JSON file from MacWhisper containing speaker labels and timestamps",
                    key="whisper_file"
                )
                if whisper_file:
                    log_message(f"Whisper file detected: {whisper_file.name}")
            
            with col2:
                wav_file = st.file_uploader(
                    "Upload WAV file",
                    type=['wav'],
                    help="The WAV file containing the audio (must match the MacWhisper file)",
                    key="wav_file"
                )
                if wav_file:
                    log_message(f"WAV file detected: {wav_file.name}")
            
            if whisper_file and wav_file and not st.session_state.get('processing', False):
                log_message(f"File upload block triggered: {whisper_file.name}, {wav_file.name}")
                st.session_state.processing = True
                
                # Save uploaded files to temporary files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_whisper:
                    tmp_whisper.write(whisper_file.getvalue())
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
                        tmp_wav.write(wav_file.getvalue())
                        
                        try:
                            log_message("Starting audio processing...")
                            
                            # Show processing status
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process large audio file in chunks
                            audio = None
                            sr = None
                            for progress, result in process_large_audio(tmp_wav.name, self.chunk_size):
                                progress_bar.progress(progress)
                                if isinstance(result, tuple):
                                    audio, sr = result
                                    status_text.text("Processing complete!")
                                else:
                                    status_text.text(result)
                            
                            if audio is None or sr is None:
                                raise Exception("Failed to process audio file")
                            
                            if self.config.get('NORMALIZE_AUDIO'):
                                log_message("Normalizing audio...")
                                audio = self.preprocessor.normalize_audio(audio)
                                log_message("Audio normalized")
                            
                            if self.config.get('REMOVE_SILENCE'):
                                log_message("Removing silence...")
                                audio = self.preprocessor.remove_silence(audio)
                                log_message("Silence removed")
                            
                            st.session_state.audio_data = audio
                            st.session_state.sample_rate = sr
                            
                            # Extract segments from MacWhisper file
                            log_message("Reading MacWhisper data...")
                            segments = self.library_builder.extract_segments_from_whisper(tmp_whisper.name)
                            log_message(f"Extracted {len(segments)} segments from whisper file.")
                            if segments:
                                log_message(f"Sample segment: {segments[0]}")
                            st.session_state.segments = segments
                            
                            # Display segments
                            st.subheader("Diarized Segments")
                            for i, segment in enumerate(segments):
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col1:
                                    st.write(f"Speaker {segment['speaker']}")
                                
                                with col2:
                                    st.write(f"{segment['start']:.2f}s - {segment['end']:.2f}s")
                                
                                with col3:
                                    if st.button("Play", key=f"play_{i}"):
                                        # Extract and play segment
                                        start_sample = int(segment['start'] * sr)
                                        end_sample = int(segment['end'] * sr)
                                        segment_audio = audio[start_sample:end_sample]
                                        
                                        # Save segment to temporary file and play
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                            sf.write(tmp.name, segment_audio, sr)
                                            st.audio(tmp.name)
                        
                        except Exception as e:
                            log_message(f"Error processing files: {str(e)}")
                            log_message(traceback.format_exc())
                            st.error(f"Error processing files: {str(e)}")
                        
                        finally:
                            # Clean up temporary files
                            os.unlink(tmp_whisper.name)
                            os.unlink(tmp_wav.name)
                            log_message("Temporary files cleaned up")
                            st.session_state.processing = False
            
            # Settings
            st.sidebar.subheader("Settings")
            similarity_threshold = st.sidebar.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=self.config.get('SIMILARITY_THRESHOLD'),
                step=0.05
            )
            self.matcher.set_similarity_threshold(similarity_threshold)
            
            normalize_audio = st.sidebar.checkbox(
                "Normalize Audio",
                value=self.config.get('NORMALIZE_AUDIO')
            )
            self.config.set('NORMALIZE_AUDIO', normalize_audio)
            
            remove_silence = st.sidebar.checkbox(
                "Remove Silence",
                value=self.config.get('REMOVE_SILENCE')
            )
            self.config.set('REMOVE_SILENCE', remove_silence)
            
        except Exception as e:
            log_message(f"Unexpected error in app: {str(e)}")
            log_message(traceback.format_exc())
            st.error(f"Unexpected error in app: {str(e)}")

def main():
    """Main entry point for the application."""
    try:
        st.set_page_config(
            page_title="Voice Diarization and Identification",
            page_icon="🎙️",
            layout="wide"
        )
        
        log_message("Starting application...")
        app = VoiceDiarizationApp()
        app.run()
        
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        log_message(f"Critical error: {str(e)}")
        log_message(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main() 