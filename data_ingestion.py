import os
import sys
import subprocess
import logging
from typing import List

# Third-party libraries
import jieba  # Keep this if it's needed by downstream scripts
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Configure logging format
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Add common module path
sys.path.append("./common")
from func_gpt import load_llm, load_retriever, llmresponse  # Ensure the path is correct

# Folders to be cleared before ingestion
FOLDERS_TO_CLEAR = [
    'docs/cleaned_json',
    'docs/cleaned_txt',
    'docs/docx',
    'docs/json',
]

# Scripts to be executed in order
SCRIPTS_TO_RUN = [
    'common/clean_data.py',
    'common/txt2json.py',
    'common/write_abstract.py',
    'common/vector_store.py',
]


def clear_folders(folders: List[str]):
    """Delete all files in the specified folders."""
    for folder in folders:
        if not os.path.exists(folder):
            logging.warning(f"Folder not found: {folder}")
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}: {e}")


def run_scripts(scripts: List[str]):
    """Run each script in the given list sequentially."""
    for script in scripts:
        try:
            logging.info(f"Running script: {script}")
            result = subprocess.run(['python', script], check=True, capture_output=True, text=True)
            logging.info(f"Output of {script}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error in {script}:\n{e.stderr}")


def main():
    """Main entry point for the data ingestion process."""
    logging.info("Starting data ingestion process...")
    clear_folders(FOLDERS_TO_CLEAR)
    run_scripts(SCRIPTS_TO_RUN)
    logging.info("Data ingestion completed.")


if __name__ == '__main__':
    main()