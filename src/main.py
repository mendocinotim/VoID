import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ui.app import main

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("HF_AUTH_TOKEN"):
        print("Error: HF_AUTH_TOKEN environment variable not set")
        print("Please set your HuggingFace authentication token:")
        print("export HF_AUTH_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Run the application
    main() 