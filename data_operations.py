import os
import cv2
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa


def resize_dataset(dataset: pd.DataFrame,
                   size: (int, int),
                   name_column: str,
                   dest_folder: str,
                   ori_folder: str = "") -> None:
    for file in dataset[name_column].values:
        img = cv2.imread(os.path.join(ori_folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        cv2.imwrite(os.path.join(dest_folder, file.split("/")[-1]), img)


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


def train_augmentation() -> iaa.Augmenter:
    seq = iaa.Sequential(
        [
            # iaa.Multiply(mul=(0.8, 1.5)),
            # sometimes(iaa.GammaContrast()),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes((iaa.pillike.Autocontrast())),
            sometimes(iaa.Crop(px=(0, 25), keep_size=True))
        ],
        random_order=True
    )
    return seq


def offline_augmentation() -> None:
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 1),
                       [
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    path_to_project = os.getcwd()
    path_to_data = os.path.join(path_to_project, "data")
    path_to_experiment = os.path.join(path_to_project, "experiments")
    path_to_image = os.path.join(path_to_data, "images")

    train_df = pd.read_csv(os.path.join(path_to_data, "train.csv"))

    if not os.path.exists(os.path.join(path_to_experiment, "dataAug")):
        os.mkdir(os.path.join(path_to_experiment, "dataAug"))

    train_true = train_df[train_df["pneumonia"] == 1]
    print(train_true.shape)
    train_false = train_df[train_df["pneumonia"] == 0]
    print(train_false.shape)

    i = 0
    train_new = "fileName,pneumonia\n"
    while i < train_false.shape[0]:
        line = train_false.iloc[[i]]
        file = line["fileName"].values[0]
        img = cv2.imread(os.path.join(path_to_image, file), cv2.IMREAD_GRAYSCALE)
        new_img = seq(image=img)
        cv2.imwrite(os.path.join(path_to_experiment, "dataAug", "aug_" + file),
                    new_img)
        train_new += "aug_" + file + ",0\n"
        i += 1

    file = open(os.path.join(path_to_experiment, "train_new.csv"), "w")
    file.write(train_new)
    file.close()
