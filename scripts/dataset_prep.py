from utils.data.csv_parsing import load_csv_as_dataframe
import os
import pandas as pd

misinformation_dict = {True: "Fake", False: "Credible"}
veracity_dict = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "mostly-false": 0,
    "false": 0,
    "pants-fire": 0,
}


def transform_fakes_dataset(dataset_path):
    print("Transforming FA-KES")
    transformed_path = os.getcwd() + "/data/transformed/"
    df = load_csv_as_dataframe(dataset_path)
    df["label"] = df["labels"]
    df.to_csv(transformed_path + "FA-KES.csv", index=False)


def tranform_recovery_news_dataset(dataset_path):
    print("Transforming recovery-news-data")
    transformed_path = os.getcwd() + "/data/transformed/"
    df = load_csv_as_dataframe(dataset_path)
    df.dropna(subset=["reliability", "body_text"], inplace=True)
    df = df[df["reliability"].isin(["0", "1"])]
    df["label"] = df["reliability"].astype(int)
    df["article_title"] = df["title"]
    df["article_content"] = df["body_text"]
    df.to_csv(transformed_path + "recovery-news-data.csv", index=False)


def transform_politifact_dataset(dataset_path):
    print("Transforming politifact")
    transformed_path = os.getcwd() + "/data/transformed/"
    df = pd.read_json(dataset_path)
    df["label"] = df["verdict"].apply(lambda x: veracity_dict[x])
    df["article_title"] = ""
    df["article_content"] = df["statement"]
    df.to_csv(transformed_path + "politifact.csv", index=False)


def transform_datasets():
    raw_path = os.getcwd() + "/data/raw/"

    transform_fakes_dataset(raw_path + "FA-KES.csv")
    tranform_recovery_news_dataset(raw_path + "recovery-news-data.csv")
    transform_politifact_dataset(raw_path + "politifact_factcheck_data.json")


# Main script to run the loading process
if __name__ == "__main__":
    transform_datasets()
