import datetime
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import pre_process
import cv2
import glob
import numpy as np
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_operations import train_augmentation
from train_data_loader import ImageDataLoader
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_generator import model_generator
from test_data_loader import TestDataLoader


def get_file_path(file_list, sop_id):
    return next((x for x in file_list if sop_id in x), '')


def do_train(is_kaggle=False,
             batch=1,
             size=(512, 512),
             shape=(512, 512, 3),
             do_pre_process=False):
    print("Iniciando treinamento")
    # Configs of the training

    initial_layers_freezed = 0
    freeze_base_model = True

    # is_kaggle = True if len(sys.argv) > 1 else False
    if not is_kaggle:
        path_to_project = os.getcwd()
        path_to_output = os.getcwd()
        path_to_data = os.path.join(path_to_project, "data")
    else:
        path_to_project = "../input/rsna-str-pulmonary-embolism-detection/"
        path_to_output = "."
        path_to_data = path_to_project

    path_to_model = os.path.join(path_to_output, "model")
    path_to_experiment = os.path.join(path_to_output, "experiments")

    train_df = pd.read_csv(os.path.join(path_to_data, "train.csv"))
    if is_kaggle:
        print("buscando arquivos")
        file_list = glob.glob("../input/rsna-str-pe-detection-jpeg-256/train-jpegs/*/*/*.jpg")
        print("mapeando localização das imagens")
        train_df["pre_processed_file"] = train_df.apply(lambda x:
                                                        get_file_path(file_list, x["SOPInstanceUID"]),
                                                        axis=1)
    else:
        train_df["file_path"] = train_df.apply(lambda x:
                                               os.path.join(path_to_data,
                                                            "train" if is_kaggle else "",
                                                            x["StudyInstanceUID"] if is_kaggle else "",
                                                            x["SeriesInstanceUID"] if is_kaggle else "",
                                                            f'{x["SOPInstanceUID"]}.dcm'),
                                               axis=1)
    print(train_df.shape)

    if is_kaggle:
        print("Carregando dataset de teste")
        test_df = pd.read_csv(os.path.join(path_to_data, "test.csv"))
        test_df["file_path"] = test_df.apply(lambda x:
                                             os.path.join(path_to_data,
                                                          "test" if is_kaggle else "",
                                                          x["StudyInstanceUID"] if is_kaggle else "",
                                                          x["SeriesInstanceUID"] if is_kaggle else "",
                                                          f'{x["SOPInstanceUID"]}.dcm'),
                                             axis=1)
        print(test_df.shape)

    if do_pre_process:
        print("Iniciando pre-processamento")
        path_to_prepress = os.path.join(path_to_experiment, "pre_process")
        if not os.path.exists(path_to_prepress):
            print("Criando pasta de destino")
            os.makedirs(path_to_prepress)
            print("Realizando pre-processamento")
            for file, name in zip(train_df["file_path"].values, train_df["SOPInstanceUID"].values):
                image = pre_process.dicom_to_jpg(file, size=size)
                cv2.imwrite(os.path.join(path_to_prepress, f"{name}.jpg"), image)

        train_df["pre_processed_file"] = train_df["SOPInstanceUID"].map(lambda x: os.path.join(path_to_prepress,
                                                                                               f"{x}.jpg"))
    if is_kaggle:
        print("Realizando split dos dados")
        unique_df = train_df[["StudyInstanceUID", "pe_present_on_image"]]
        unique_df = unique_df.groupby(by=["StudyInstanceUID"]).sum()
        unique_df["pe_bool"] = np.where(unique_df["pe_present_on_image"] > 0, True, False)


        train_to_merge, valid_to_merge = train_test_split(unique_df,
                                              test_size=0.25,
                                              random_state=42,
                                              stratify=unique_df[["pe_bool"]])

        print("Realizando merge apos estratificação")
        train_df = pd.merge(left=train_df, right=train_to_merge, how="inner", on="StudyInstanceUID")
        valid_df = pd.merge(left=train_df, right=valid_to_merge, how="inner", on="StudyInstanceUID")


    else:
        valid_df = train_df.iloc[-1:]
        train_df = train_df.iloc[:-1]

    print("train", train_df.shape[0], "validation", valid_df.shape[0])

    data_execucao = datetime.datetime.today().strftime("%d-%m_%H-%M")

    os.makedirs(os.path.join(path_to_model, data_execucao))
    path_to_result = os.path.join(path_to_model, data_execucao)

    weight_file = "weights.h5"
    weights_filepath = os.path.join(path_to_result, "" + weight_file)

    print("Carregando modelo")
    net = model_generator(batch=batch,
                          size=size,
                          shape=shape,
                          path_to_image=path_to_data,
                          path_to_save=path_to_result)

    model = net.get_model(freeze_base_model, initial_layers_freezed)
    print(model.summary())

    train_data = ImageDataLoader(df=train_df,
                                 images_dir_path="",
                                 image_size=size,
                                 image_filename_column="pre_processed_file",
                                 label_column=["pe_present_on_image",
                                               "rv_lv_ratio_gte_1",
                                               "rv_lv_ratio_lt_1",
                                               "leftsided_pe",
                                               "chronic_pe",
                                               "rightsided_pe",
                                               "acute_and_chronic_pe",
                                               "central_pe",
                                               "indeterminate"], augmenter=None)
    # augmenter=train_augmentation())

    train_loader = train_data.to_tf_dataset(batch_size=batch)
    val_data = ImageDataLoader(df=valid_df, images_dir_path="", image_size=size,
                               image_filename_column="pre_processed_file",
                               label_column=["pe_present_on_image",
                                             "rv_lv_ratio_gte_1",
                                             "rv_lv_ratio_lt_1",
                                             "leftsided_pe",
                                             "chronic_pe",
                                             "rightsided_pe",
                                             "acute_and_chronic_pe",
                                             "central_pe",
                                             "indeterminate"], augmenter=None)
    val_loader = val_data.to_tf_dataset(batch_size=batch)

    net.save_model_as_json()
    #
    callbacks = [ModelCheckpoint(weights_filepath, monitor='val_loss', mode='max',
                                 verbose=1, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=12, mode="max"),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4,
                                   verbose=0, mode='max', min_delta=0.0001,
                                   cooldown=0, min_lr=0)]

    steps = int(train_df.shape[0] / batch)
    val_steps = int(valid_df.shape[0] / batch)

    history = model.fit(train_loader,
                        epochs=100,
                        verbose=1,
                        validation_data=val_loader,
                        callbacks=callbacks,
                        steps_per_epoch=steps,
                        validation_steps=val_steps)

    # Plot train graphs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.legend(['treino', 'validação'], loc='upper left')
    plt.savefig(os.path.join(path_to_result, "loss.png"))
    plt.clf()

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Época')
    # plt.legend(['treino', 'validação'], loc='upper left')
    # plt.savefig(os.path.join(path_to_result, "acc.png"))

    # Load Model
    # json_file = open(os.path.join(path_to_result, "model.json"), 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # net.load_model_from_json(loaded_model_json, os.path.join(path_to_result, "weights.h5"))
    #
    # data_for_validation = val_data.to_tf_dataset(batch_size=1, shuffle=False)
    # real_values = valid_df["pe_present_on_image",
    #                        "rv_lv_ratio_gte_1",
    #                        "rv_lv_ratio_lt_1",
    #                        "leftsided_pe",
    #                        "chronic_pe",
    #                        "rightsided_pe",
    #                        "acute_and_chronic_pe",
    #                        "central_pe",
    #                        "indeterminate"].to_dict("list")
    # # .values
    #
    # predictions = model.predict(data_for_validation, steps=valid_df.shape[0], verbose=1)
    # predictions = [0 if x < 0.5 else 1 for x in predictions]
    #
    # accuracy = accuracy_score(real_values, predictions)
    # print("Test Accuracy:", accuracy)
    # f1 = f1_score(real_values, predictions)
    # print("F1 score:", f1)
    #
    # confusion_mtx = confusion_matrix(real_values, predictions)
    #
    # # ax = plt.axes()
    # # sn.heatmap(confusion_mtx, annot=True, annot_kws={"size": 25}, cmap="Blues", ax=ax)
    # # ax.set_title('Test Accuracy', size=14)
    # # plt.show()
    #
    # # test_data = TestDataLoader(df=test_df,
    # #                             images_dir_path="",
    # #                             image_size=size,
    # #                             image_filename_column="file_path")
    # # test_loader = test_data.to_tf_dataset(batch_size=1)
    # # test_pred = model.predict(test_loader, steps=test_df.shape[0], verbose=1)
    # # test_pred = [0 if x < 0.5 else 1 for x in test_pred]
    # # filenames = test_df['fileName']
    # # results = pd.DataFrame({"fileName": filenames,
    # #                         "pneumonia": test_pred})
    # # results.to_csv(os.path.join(path_to_result, "results.csv"), index=False)
    #
    # # Save train configs
    # descricao_treino = f"""CNN Model: Xception
    #                        Batch: {batch}
    #                        Metric: Acurracy
    #                        Loss: Binary_CrossEntropy
    #                        Last layer: Many Sigmoidal
    #                        Data augmentation: -
    #                        Image resolution : {size}
    #                        Image shape : {shape}
    #                        Freezed layers: {initial_layers_freezed}
    #                        Stratify: True
    #                        Freeze Base Model: {freeze_base_model}
    #                     """
    # file = open(os.path.join(path_to_result, "train_description.txt", "w"))
    # file.write(os.path.join(path_to_result, descricao_treino))
    # file.close()
    #
    # files = []
    # for (_, _, filenames) in os.walk(path_to_result):
    #     files.extend(filenames)
    #     break
    # zip_file = zipfile.ZipFile(f"{data_execucao}.zip", 'w')
    # with zip_file:
    #     for file in files:
    #         zip_file.write(os.path.join(path_to_result, file))


if __name__ == "__main__":
    do_train()
