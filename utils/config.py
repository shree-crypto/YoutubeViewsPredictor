"""
Configuration Management Module

Loads and manages application configuration from YAML file.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the application."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration."""
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, looks for config.yaml in root.
        """
        if config_path is None:
            # Look for config.yaml in project root
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / 'config.yaml'
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = self._get_default_config()
            return
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'model': {
                'type': 'xgboost',
                'random_state': 42,
                'test_size': 0.2,
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_jobs': -1
                }
            },
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'sample_size': 1000
            },
            'persistence': {
                'model_dir': 'models',
                'model_file': '{model_type}_model.joblib',
                'scaler_file': 'scaler.joblib',
                'config_file': 'model_config.json'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config
