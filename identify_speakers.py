from src.voiceprint.library_builder import VoiceprintLibraryBuilder
from src.voiceprint.database import VoiceprintDatabase
from src.voiceprint.matcher import VoiceprintMatcher
from src.voiceprint.extractor import VoiceprintExtractor
from src.diarization.diarizer import Diarizer
from src.config.settings import Config
from src.audio.audio_preprocessing import AudioPreprocessor
import soundfile as sf
import numpy as np
import pandas as pd
import os
import time

# Path to the audio file to process
AUDIO_FILE = "/Volumes/AI_ETS_2TB/RawInterviews_forTranscription/Raw SwingThing Interviews/1st Batch received on 11:26:2019/wav files/3.wav"

# Path to the voiceprint DB (adjust if needed)
DB_PATH = os.path.expanduser("~/.voice_diarization_db")

# Read Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")

# Initialize components
config = Config()
preprocessor = AudioPreprocessor()
extractor = VoiceprintExtractor(embedding_size=config.get('EMBEDDING_SIZE'))
database = VoiceprintDatabase(db_path=DB_PATH)
matcher = VoiceprintMatcher(similarity_threshold=config.get('SIMILARITY_THRESHOLD'), extractor=extractor)
diarizer = Diarizer(auth_token=HF_TOKEN)

# Load audio
print(f"[{time.ctime()}] Loading audio file: {AUDIO_FILE}")
audio, sr = sf.read(AUDIO_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # Convert to mono if needed
print(f"[{time.ctime()}] Audio loaded. Duration: {len(audio)/sr:.2f} seconds, Sample rate: {sr}")

total_duration = len(audio) / sr
chunk_size = 30  # seconds
num_chunks = int(np.ceil(total_duration / chunk_size))

csv_path = "identified_segments_3.csv"
all_results = []

print(f"[{time.ctime()}] Starting chunked diarization and identification...")
for chunk_idx in range(num_chunks):
    chunk_start = chunk_idx * chunk_size
    chunk_end = min((chunk_idx + 1) * chunk_size, total_duration)
    print(f"[{time.ctime()}] Processing chunk {chunk_idx+1}/{num_chunks}: {chunk_start:.2f}s - {chunk_end:.2f}s")
    chunk_audio = audio[int(chunk_start*sr):int(chunk_end*sr)]
    if len(chunk_audio) == 0:
        continue
    try:
        print(f"[{time.ctime()}]  Diarizing chunk...")
        diari_start = time.time()
        segments = diarizer.diarize(chunk_audio, sr)
        diari_end = time.time()
        print(f"[{time.ctime()}]  Diarization complete. Found {len(segments)} segments. Took {diari_end-diari_start:.2f} seconds.")
        # Adjust segment times to global audio
        for i, seg in enumerate(segments):
            start = seg['start'] + chunk_start
            end = seg['end'] + chunk_start
            print(f"[{time.ctime()}]   Processing segment {i+1}/{len(segments)}: {start:.2f}s - {end:.2f}s")
            seg_audio = audio[int(start*sr):int(end*sr)]
            print(f"[{time.ctime()}]   Extracting voiceprint...")
            features = preprocessor.extract_features(seg_audio, sr, n_mfcc=40)
            embedding = extractor.extract_embedding(features)
            print(f"[{time.ctime()}]   Matching to known speakers...")
            speaker, confidence = matcher.find_best_match(embedding, database.get_voiceprints())
            if not speaker or confidence < matcher.similarity_threshold:
                speaker = f"Unknown (best: {speaker or 'None'})"
            print(f"[{time.ctime()}]   Segment assigned to: {speaker} (confidence: {confidence:.3f})")
            all_results.append({
                'start': start,
                'end': end,
                'assigned_speaker': speaker,
                'confidence': confidence
            })
        # Save after each chunk
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"[{time.ctime()}]  Partial results saved to {csv_path}")
    except Exception as e:
        print(f"[{time.ctime()}]  Error processing chunk {chunk_idx+1}: {e}")
        continue

print(f"\n[{time.ctime()}] Chunked diarization and identification complete.")
print(f"[{time.ctime()}] Final results saved to {csv_path}") 