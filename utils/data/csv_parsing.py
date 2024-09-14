import csv
import chardet
import pandas as pd


def detect_encoding(csv_file_path):
    with open(csv_file_path, "rb") as f:
        result = chardet.detect(f.read())
        encoding = result["encoding"]
    return encoding


def load_csv_as_dicts(csv_file_path):
    # Detect the file encoding
    encoding = detect_encoding(csv_file_path=csv_file_path)

    # Read the CSV file with the detected encoding
    with open(csv_file_path, newline="", encoding=encoding) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        data = [row for row in csv_reader]
    return data


def load_csv_as_dataframe(csv_file_path):
    # Detect the file's encoding
    encoding = detect_encoding(csv_file_path=csv_file_path)

    # Read the CSV using the detected encoding
    df = pd.read_csv(csv_file_path, encoding=encoding)
    return df
