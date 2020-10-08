import os
import pandas as pd
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np

path_to_project = os.getcwd()
path_to_data = os.path.join(path_to_project, "data")
path_to_model = os.path.join(path_to_project, "model")
path_to_experiment = os.path.join(path_to_project, "experiments")
path_to_image = os.path.join(path_to_data, "images")

def resize_dataset(dataset: pd.DataFrame,
                   size: (int, int),
                   name_column: str,
                   dest_folder: str,
                   ori_folder: str = "") -> None:
    for file in dataset[name_column].values:
        img = cv2.imread(os.path.join(ori_folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imgOri = np.copy(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        equ = cv2.equalizeHist(imgOri)
        res = np.hstack((imgOri, equ, img))
        cv2.imwrite(os.path.join(dest_folder, file.split("/")[-1]), res)


train_df = pd.read_csv(os.path.join(path_to_data, "train.csv"))
test_df = pd.read_csv(os.path.join(path_to_data, "test.csv"))

if not os.path.exists(os.path.join(path_to_experiment, "train", "0")):
    os.makedirs(os.path.join(path_to_experiment, "train", "0"))
    os.makedirs(os.path.join(path_to_experiment, "train", "1"))
    os.makedirs(os.path.join(path_to_experiment, "test"))

if not os.path.exists(os.path.join(path_to_experiment, "contr")):
    os.makedirs(os.path.join(path_to_experiment, "contr"))
    resize_dataset(dataset=train_df,
                   size=(224,224),
                   name_column="fileName",
                   dest_folder=os.path.join(path_to_experiment, "contr"),
                   ori_folder=path_to_image)
    resize_dataset(dataset=test_df,
                   size=(224, 224),
                   name_column="fileName",
                   dest_folder=os.path.join(path_to_experiment, "contr"),
                   ori_folder=path_to_image)



asp_ratio_train = []
asp_ratio_test = []
for fname, pneumonia in train_df[:].values:
    shape = cv2.imread(os.path.join(path_to_image, fname)).shape
    asp_ratio_train.append(shape[0] / shape[1])
    if not os.path.exists(os.path.join(path_to_experiment, "train", str(pneumonia), fname)):
        shutil.copy2(os.path.join(path_to_image, fname),
                     os.path.join(path_to_experiment, "train", str(pneumonia), fname))

for fname in test_df["fileName"].values:
    shape = cv2.imread(os.path.join(path_to_image, fname)).shape
    asp_ratio_test.append(shape[0] / shape[1])
    if not os.path.exists(os.path.join(path_to_experiment, "test", fname)):
        shutil.copy2(os.path.join(path_to_image, fname),
                     os.path.join(path_to_experiment, "test", fname))

plt.hist(asp_ratio_train)
plt.savefig(os.path.join(path_to_experiment, "train_ratio.png"))
plt.clf()

plt.hist(asp_ratio_test)
plt.savefig(os.path.join(path_to_experiment, "test_ratio.png"))
