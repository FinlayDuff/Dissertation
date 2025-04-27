import csv
import chardet
import pandas as pd


def detect_encoding(path, sample_size=100_000):
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    result = chardet.detect(raw)
    enc = result["encoding"] or "utf-8"
    return enc, result.get("confidence", 0)


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
