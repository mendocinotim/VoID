import os
import json
import re
from collections import defaultdict

# Configuration
ROOT_DIR = "/Volumes/AI_ETS_2TB/RawInterviews_forTranscription/Transcribed Raw Interviews/transcriptions"
TRANSCRIPT_EXTENSIONS = ('.json', '.js')
GENERIC_LABEL_PATTERN = re.compile(r"speaker ?\d+", re.IGNORECASE)

# Helper to extract speaker labels from a transcript file
def extract_speaker_labels(transcript_path):
    speakers = set()
    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        # Try to find speaker labels in common transcript formats
        if isinstance(data, dict):
            # MacWhisper format: segments with 'speaker' field
            if 'segments' in data:
                for seg in data['segments']:
                    if 'speaker' in seg:
                        speakers.add(str(seg['speaker']))
            # Other possible formats
            if 'speakers' in data and isinstance(data['speakers'], list):
                for s in data['speakers']:
                    speakers.add(str(s))
        elif isinstance(data, list):
            for seg in data:
                if isinstance(seg, dict) and 'speaker' in seg:
                    speakers.add(str(seg['speaker']))
    except Exception as e:
        print(f"Error reading {transcript_path}: {e}")
    return speakers

def main():
    label_usage = defaultdict(list)  # label -> list of (dir, file)
    all_labels = set()
    for dir_name in os.listdir(ROOT_DIR):
        dir_path = os.path.join(ROOT_DIR, dir_name)
        if not os.path.isdir(dir_path) or dir_name.startswith('.'):
            continue
        json_dir = os.path.join(dir_path, 'JSON')
        if not os.path.exists(json_dir):
            continue
        for fname in os.listdir(json_dir):
            if fname.endswith(TRANSCRIPT_EXTENSIONS):
                transcript_path = os.path.join(json_dir, fname)
                speakers = extract_speaker_labels(transcript_path)
                for label in speakers:
                    label_usage[label].append((dir_name, fname))
                    all_labels.add(label)
    # Print summary
    print("\n=== Speaker Label Usage Summary ===")
    for label in sorted(all_labels):
        flag = ""
        if GENERIC_LABEL_PATTERN.match(label):
            flag = "[GENERIC]"
        elif label.strip().lower() == "unknown":
            flag = "[UNKNOWN]"
        elif label.replace(' ', '').lower() in [l.replace(' ', '').lower() for l in all_labels if l != label]:
            flag = "[INCONSISTENT?]"
        print(f"\nLabel: {label} {flag}")
        for dir_name, fname in label_usage[label]:
            print(f"  - {dir_name}/JSON/{fname}")
    print("\nDone. Review the above for generic, unknown, or inconsistent labels.")

if __name__ == "__main__":
    main() 