import pandas as pd
import numpy as np
from PIL import Image, ImageStat
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser("Pre-processing script that removes grayscale images and split the dataset into train, validation and test")
parser.add_argument("--dataset_path", type=str, default="C:/Users/loren/Datasets/Places205/", help="Absolute path to the Places205 dataset, eg: /../../Places205/")
parser.add_argument("--train_split", type=float, default=0.85, help="Percentage of the dataset to be used for training [0,1]")
parser.add_argument("--val_split", type=float, default=0.05, help="Percentage of the dataset to be used for validation [0,1]")
parser.add_argument("--test_split", type=float, default=0.1, help="Percentage of the dataset to be used for testing [0,1]")

args = parser.parse_args()

DATASET_PATH = args.dataset_path
TRAIN_SPLIT = args.train_split
VAL_SPLIT = args.val_split
TEST_SPLIT = args.test_split

# set seed for reproducibility (shuffle operation later)
SEED = 123456
np.random.seed(SEED)

df = pd.read_csv(DATASET_PATH + "files.csv")

# check which images are in grayscale and add a corresponding column to the dataframe
def is_grayscale(path):
    img = Image.open(DATASET_PATH + path)
    is_L_mode = img.mode == "L"
    img = img.convert("RGB")
    stat = ImageStat.Stat(img)
    img.close()
    return path, sum(stat.sum) / 3 == stat.sum[0] or is_L_mode


print("Checking which images are in grayscale...")
results = []
with ThreadPoolExecutor() as executor:
    paths = df["file_name"].tolist()
    # better to use a set since it's faster to check if an element is in a set (constant time) than in a list
    futures = {executor.submit(is_grayscale, path) for path in paths}
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

df = df.merge(pd.DataFrame(results, columns=["file_name", "grayscale"]), on="file_name", how="left")

n_grayscale = df["grayscale"].sum()
n_total = len(df)
print(f"Number of grayscale images: {n_grayscale}/{n_total} ({n_grayscale/n_total*100:.2f}%)")

# remove grayscale images and compute the minimum number of samples across all classes
df = df[df["grayscale"] == False]
df = df.drop(columns=["grayscale"])

# split the dataset into train, validation and test
train_test_ss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
train_val_ss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=SEED)

for train_index, test_index in train_test_ss.split(df, df['class']):
    train_val_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

for train_index, val_index in train_val_ss.split(train_val_df, train_val_df['class']):
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat([train_df, val_df, test_df])

print(f"Number of samples in train: {len(train_df)}")
print(f"Number of samples in val: {len(val_df)}")
print(f"Number of samples in test: {len(test_df)}")

print("Shuffling and splitting done, saving to file...")
df.to_csv(DATASET_PATH + "files_split.csv", index=False)
print("Done")

