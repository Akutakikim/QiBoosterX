# quantum_data_loader.py

import os
import json
import pandas as pd

class QuantumDataLoader:
    """
    QuantumDataLoader provides static methods to load text, JSON, and CSV datasets 
    for quantum AI operations in QiBoosterX.
    """

    @staticmethod
    def load_text_data(file_path, delimiter="\n", auto_delimiter=False):
        """
        Load text data from a plain file. Each segment split by the delimiter is treated as one entry.
        
        Args:
            file_path (str): Path to the text file.
            delimiter (str): Delimiter to split data (default: newline).
            auto_delimiter (bool): If True, auto-detects delimiter by file content.
        
        Returns:
            list[str]: List of text entries.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[QiBoosterX][Error] Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            
            if auto_delimiter:
                if "\n" in content:
                    delimiter = "\n"
                elif "\t" in content:
                    delimiter = "\t"
                elif "," in content:
                    delimiter = ","
                else:
                    delimiter = " "  # fallback to space if unsure

            return content.split(delimiter)

    @staticmethod
    def load_json_data(file_path):
        """
        Load structured data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            dict | list: Parsed JSON object.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[QiBoosterX][Error] JSON file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"[QiBoosterX][Error] Invalid JSON format in {file_path}: {e}")

    @staticmethod
    def load_csv_data(file_path, delimiter=","):
        """
        Load structured tabular data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            delimiter (str): Delimiter for the CSV file (default: comma).
        
        Returns:
            pd.DataFrame: DataFrame containing loaded data.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[QiBoosterX][Error] CSV file not found: {file_path}")
        
        try:
            return pd.read_csv(file_path, delimiter=delimiter)
        except pd.errors.ParserError as e:
            raise ValueError(f"[QiBoosterX][Error] Error parsing CSV at {file_path}: {e}")