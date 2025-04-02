import os
import numpy as np
from data.utils import *

data_folder = "data/cifar-10-batches-py/"

batch_info = unpickle(os.path.join(data_folder, "batches.meta"))

all_images = None
all_labels = []
all_files = []
for i in range(1, 6):
    batch_name = "data_batch_" + str(i)
    batch = unpickle(os.path.join(data_folder, batch_name))
    all_images = (
        np.vstack((all_images, batch[b"data"]))
        if all_images is not None
        else batch[b"data"]
    )
    all_labels += batch[b"labels"]
    all_files += batch[b"filenames"]

all_files = [file.decode("utf-8") for file in all_files]

save_data(
    {
        "images": all_images,
        "labels": all_labels,
        "filenames": all_files,
        "label_names": batch_info[b"label_names"],
    },
    os.path.join(data_folder, "data.pkl"),
)