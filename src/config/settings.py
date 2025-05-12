import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # Audio processing
    'SAMPLE_RATE': 16000,
    'NORMALIZE_AUDIO': True,
    'REMOVE_SILENCE': True,
    
    # Diarization
    'MIN_SPEAKERS': None,
    'MAX_SPEAKERS': None,
    'DIARIZATION_MODEL': 'pyannote/speaker-diarization',
    
    # Voiceprint extraction
    'EMBEDDING_SIZE': 128,
    'SIMILARITY_THRESHOLD': 0.7,
    
    # Database
    'DB_PATH': os.path.join(os.path.expanduser('~'), '.voice_diarization_db'),
    
    # UI
    'UI_THEME': 'light',
    'AUTO_SAVE': True,
    'SAVE_INTERVAL': 300,  # seconds
    'CHUNK_SIZE': 300,  # 5 minutes per chunk for large file processing
}

class Config:
    """Configuration manager for the voice diarization system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to custom config file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        import json
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            self.config.update(custom_config)
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self.config.copy() 