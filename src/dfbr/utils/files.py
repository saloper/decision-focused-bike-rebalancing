"""
Module Name: files.py
Description: used to join absolute project root path with a relative path for easy I/O
Usage: from src.utils.files import get_path 
"""
import yaml
from pathlib import Path

#Go up to levels to get to the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def get_path(relative_path_str):
    return PROJECT_ROOT / relative_path_str

def get_config(config_file):
    config_path = get_path(f"configs/{config_file}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
    
    #Convert file path strings to path objects 
    if 'paths' in config:
        for key, value in config['paths'].items():
                config['paths'][key] = get_path(value)
                
    return config