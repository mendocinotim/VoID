import json
import os
from typing import List, Dict
import datetime

def whisper_to_rttm(whisper_file: str, output_file: str = None) -> str:
    """
    Convert a MacWhisper JSON file to RTTM format.
    
    Args:
        whisper_file: Path to the MacWhisper JSON file
        output_file: Path to save the RTTM file (optional)
        
    Returns:
        Path to the created RTTM file
    """
    # Read the Whisper JSON file
    with open(whisper_file, 'r', encoding='utf-8') as f:
        whisper_data = json.load(f)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(whisper_file))[0]
    
    # If no output file specified, create one in the same directory
    if output_file is None:
        output_file = os.path.join(os.path.dirname(whisper_file), f"{base_name}.rttm")
    
    # Convert segments to RTTM format
    rttm_lines = []
    for segment in whisper_data.get('segments', []):
        # Extract timing information
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time
        
        # Extract speaker information
        speaker = segment.get('speaker', 'UNKNOWN')
        
        # Create RTTM line
        # Format: SPEAKER file_name 1 start_time duration <NA> <NA> speaker <NA> <NA>
        rttm_line = f"SPEAKER {base_name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        rttm_lines.append(rttm_line)
    
    # Write RTTM file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rttm_lines))
    
    return output_file

def batch_convert_whisper_to_rttm(input_dir: str, output_dir: str = None) -> List[str]:
    """
    Convert all MacWhisper JSON files in a directory to RTTM format.
    
    Args:
        input_dir: Directory containing MacWhisper JSON files
        output_dir: Directory to save RTTM files (optional)
        
    Returns:
        List of paths to created RTTM files
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    converted_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.rttm")
            
            try:
                rttm_file = whisper_to_rttm(input_path, output_path)
                converted_files.append(rttm_file)
                print(f"Converted {filename} to RTTM format")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")
    
    return converted_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MacWhisper files to RTTM format')
    parser.add_argument('input', help='Input MacWhisper JSON file or directory')
    parser.add_argument('--output', help='Output RTTM file or directory (optional)')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_convert_whisper_to_rttm(args.input, args.output)
    else:
        whisper_to_rttm(args.input, args.output) 