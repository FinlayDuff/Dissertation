from utils.data.csv_parsing import load_csv_as_dataframe
import os

misinformation_dict = {True: "Fake", False: "Credible"}


def transform_fakes_dataset(dataset_path):
    transformed_path = os.getcwd() + "/data/transformed/"
    df = load_csv_as_dataframe(dataset_path)
    df["label"] = df["labels"]  # .apply(lambda x: misinformation_dict[not bool(x)])
    df.to_csv(transformed_path + "FA-KES.csv", index=False)


def transform_datasets():
    raw_path = os.getcwd() + "/data/raw/"

    print("Transforming FA-KES")
    transform_fakes_dataset(raw_path + "FA-KES.csv")


# Main script to run the loading process
if __name__ == "__main__":
    transform_datasets()
