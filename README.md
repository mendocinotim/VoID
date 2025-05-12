# Voice Identification System (voID)

A system for identifying known voices in audio files using MacWhisper transcriptions and voiceprint matching.

## Features

- Import known voices from MacWhisper JSON files and corresponding WAV files
- Build voiceprint library from diarized audio segments
- Identify known voices in new audio files
- Interactive web interface using Streamlit

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mendocinotim/voID.git
cd voID
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face token:
   - Create a `.streamlit/secrets.toml` file
   - Add your Hugging Face token:
   ```toml
   HF_AUTH_TOKEN = "your_token_here"
   ```

## Usage

1. Start the application:
```bash
streamlit run src/ui/voice_identifier.py
```

2. Add known voices:
   - Upload MacWhisper JSON file (contains speaker labels and timestamps)
   - Upload corresponding WAV file
   - Enter a name for the voice

3. Process new audio:
   - Upload MacWhisper JSON file
   - Upload corresponding WAV file
   - View identification results

## Project Structure

```
voID/
├── src/
│   ├── audio/           # Audio processing modules
│   ├── diarization/     # Speaker diarization
│   ├── voiceprint/      # Voiceprint extraction and matching
│   ├── ui/             # Streamlit interface
│   └── config/         # Configuration
├── requirements.txt
└── README.md
```

## License

MIT License 