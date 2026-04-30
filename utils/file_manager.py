#!/usr/bin/env python3
"""
Utility script to manage test images between resources and data/input directories.
"""

import os
import shutil
import argparse
from pathlib import Path

def list_resources():
    """List available test images in resources directory"""
    resources_dir = Path("resources")
    heic_files = list(resources_dir.glob("IMG_*.HEIC"))
    
    print("Available test images in resources/:")
    for i, file in enumerate(heic_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
    
    return heic_files

def move_to_input(filename=None):
    """Move test images from resources to data/input for processing"""
    resources_dir = Path("resources")
    input_dir = Path("data/input")
    input_dir.mkdir(exist_ok=True)
    
    if filename:
        # Move specific file
        src = resources_dir / filename
        dst = input_dir / filename
        if src.exists():
            shutil.move(src, dst)
            print(f"Moved: {filename}")
        else:
            print(f"File not found: {filename}")
    else:
        # Move all HEIC files
        heic_files = list(resources_dir.glob("IMG_*.HEIC"))
        if not heic_files:
            print("No HEIC files found in resources/")
            return
        
        for file in heic_files:
            dst = input_dir / file.name
            if not dst.exists():  # Don't overwrite
                shutil.move(file, dst)
                print(f"Moved: {file.name}")
        
        print(f"\nMoved {len(heic_files)} files to data/input/")

def clear_input():
    """Clear all files from data/input directory"""
    input_dir = Path("data/input")
    if not input_dir.exists():
        print("data/input/ directory does not exist")
        return
    
    files = list(input_dir.glob("*"))
    if not files:
        print("data/input/ is already empty")
        return
    
    for file in files:
        if file.is_file():
            file.unlink()
            print(f"Removed: {file.name}")
    
    print(f"Cleared {len(files)} files from data/input/")

def main():
    parser = argparse.ArgumentParser(description="Manage test images for Shape Extractor")
    parser.add_argument("action", choices=["list", "move", "clear"], 
                       help="Action to perform")
    parser.add_argument("--file", help="Specific file to move (for move action)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_resources()
    elif args.action == "move":
        move_to_input(args.file)
    elif args.action == "clear":
        clear_input()

if __name__ == "__main__":
    main()
