from utils.data.csv_parsing import load_csv_as_dataframe
import os
import pandas as pd
from utils.constants import veracity_dict, misinformation_dict


def extract_few_shot_examples(df, label_col, text_col, n_per_class=2):
    # This function extracts few-shot examples from the dataset
    # and formats them into a string block for few-shot learning.
    # It also ensures that the few-shot examples are balanced across classes
    # and samples without replacement, ensuring no data leakage.
    few_shot_strings = []
    few_shot_ids = []

    # Create a stable, unique ID for each row to avoid accidental drops
    df = df.copy()
    df["__row_id__"] = df.index

    for label in df[label_col].unique():
        class_df = df[df[label_col] == label]
        sampled = class_df.sample(n=n_per_class, replace=False, random_state=42)
        few_shot_ids.extend(sampled["__row_id__"].tolist())

        for _, row in sampled.iterrows():
            article_content = row[text_col]
            example = f"""Example:
Article: {article_content}
Classification: {"Misinformation" if row[label_col] == 0 else "Not Misinformation"}
"""
            few_shot_strings.append(example)

    # Safely drop sampled few-shot rows
    df = df[~df["__row_id__"].isin(few_shot_ids)].copy()
    df.drop(columns=["__row_id__"], inplace=True)

    few_shot_block = "\n\n".join(few_shot_strings)
    return df, few_shot_block


def balanced_sample(df, label_column, total_samples):
    unique_labels = df[label_column].unique()
    samples_per_class = total_samples // len(unique_labels)
    balanced = []

    for label in unique_labels:
        class_df = df[df[label_column] == label]
        if len(class_df) < samples_per_class:
            raise ValueError(f"Not enough samples for label {label}")
        sampled = class_df.sample(n=samples_per_class, replace=False, random_state=42)
        balanced.append(sampled)

    return pd.concat(balanced).reset_index(drop=True)


def transform_dataset(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    dataset_name: str,
    total_samples: int = None,
):

    transform_dataset_name = f"{dataset_name}_{total_samples}"

    df[text_column] = df[text_column].apply(lambda x: x.strip().replace("\n", " "))

    # First we generate the few_shot examples, ensuring we remove these from the evaluation data
    df, few_shot_block = extract_few_shot_examples(df, label_column, text_column)
    few_shot_file_path = os.path.join(
        os.getcwd() + "/data/fewshot/", f"{transform_dataset_name}.txt"
    )
    with open(few_shot_file_path, "w", encoding="utf-8") as f:
        f.write(few_shot_block)
    print(f"[INFO] Few-shot block saved to: {few_shot_file_path}")

    # Then we sample the data to ensure we have a balanced dataset for testing
    if total_samples:
        df = balanced_sample(df, label_column, total_samples)

    file_path = os.path.join(
        os.getcwd() + "/data/transformed/", f"{transform_dataset_name}.csv"
    )

    df.to_csv(file_path, index=False)
    print(f"Saved transformed dataset to: {file_path}")

    return file_path


def transform_fakes_dataset(
    dataset_path,
    total_samples=None,
):
    print("Transforming FA-KES")
    df = load_csv_as_dataframe(dataset_path)
    df["label"] = df["labels"]
    return transform_dataset(
        df=df,
        text_column="article_content",
        label_column="label",
        dataset_name="FA-KES",
        total_samples=total_samples,
    )


def tranform_recovery_news_dataset(
    dataset_path,
    total_samples=None,
):
    print("Transforming recovery-news-data")
    df = load_csv_as_dataframe(dataset_path)
    df.dropna(subset=["reliability", "body_text"], inplace=True)
    df = df[df["reliability"].isin(["0", "1"])]
    df["label"] = df["reliability"].astype(int)
    df["article_title"] = df["title"]
    df["article_content"] = df["body_text"]
    return transform_dataset(
        df=df,
        text_column="article_content",
        label_column="label",
        dataset_name="recovery-news-data",
        total_samples=total_samples,
    )


def transform_politifact_dataset(
    dataset_path,
    total_samples=None,
):
    print("Transforming politifact")
    df = pd.read_json(dataset_path)
    df["label"] = df["verdict"].apply(lambda x: veracity_dict[x])
    df["article_title"] = ""
    df["article_content"] = df["statement"]
    return transform_dataset(
        df=df,
        text_column="article_content",
        label_column="label",
        dataset_name="politifact",
        total_samples=total_samples,
    )
