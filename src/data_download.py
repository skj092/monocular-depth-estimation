import os
import keras

# Define dataset download and extraction function
def download_and_extract(dataset_name):
    dataset_folder = f"/dataset/{dataset_name}"
    if not os.path.exists(os.path.abspath(".") + dataset_folder):
        dataset_zip = keras.utils.get_file(
            f"{dataset_name}.tar.gz",
            cache_subdir=os.path.abspath("."),
            origin=f"http://diode-dataset.s3.amazonaws.com/{dataset_name}.tar.gz",
            extract=True,
        )
        print(f"{dataset_name} dataset downloaded and extracted.")
    else:
        print(f"{dataset_name} dataset already exists.")

# Download 'val' and 'train' datasets
# download_and_extract("val")
download_and_extract("train")

