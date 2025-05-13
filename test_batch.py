from src.voiceprint.library_builder import VoiceprintLibraryBuilder
from src.config.settings import Config

# Set your root directory and range here
root_dir = "/Volumes/AI_ETS_2TB/RawInterviews_forTranscription/Transcribed Raw Interviews/transcriptions"
start_ref = "1_Johnny Mandel"
end_ref = "42_Steve Allen"

# Initialize config and builder
config = Config()
builder = VoiceprintLibraryBuilder(config)

# Run batch process
print(f"Processing from {start_ref} to {end_ref} in {root_dir}...")
df1, df2 = builder.batch_process_range(root_dir, start_ref, end_ref)

# Print results
print("\nTable 1: RefDir/SpeakerName")
print(df1)
print("\nTable 2: Speaker Summary")
print(df2) 