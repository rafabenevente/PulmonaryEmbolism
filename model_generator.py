import os

from tensorflow.keras import Input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.optimizers import Adam


class model_generator:
    def __init__(self, batch, size, shape, path_to_image, path_to_save):
        self.batch = batch
        self.size = size
        self.shape = shape
        self.path_to_image = path_to_image
        self.path_to_save = path_to_save
        self.model = None

    def get_model(self,freeze_base_model=False, freeze_initial_layers=0):
        # pretrained_model = Xception(weights="imagenet", include_top=False, input_shape=self.shape)
        input = Input(self.shape)
        pretrained_model = Xception(weights="imagenet", include_top=False)

        if freeze_base_model:
            pretrained_model.trainable = False
        else:
            for i, layer in enumerate(pretrained_model.layers):
                if i >= freeze_initial_layers:
                    break
                layer.trainable = False

        outputs = pretrained_model(input, training=False)
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Dropout(0.25)(outputs)
        outputs = Dense(1024, activation='relu')(outputs)
        outputs = Dense(256, activation='relu')(outputs)
        outputs = Dense(64, activation='relu')(outputs)
        ppoi = Dense(1, activation='sigmoid', name='pe_present_on_image')(outputs)
        rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)
        rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs)
        lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)
        cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)
        rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)
        aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)
        cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)
        indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)

        self.model = Model(inputs=input, outputs={'pe_present_on_image': ppoi,
                                                  'rv_lv_ratio_gte_1': rlrg1,
                                                  'rv_lv_ratio_lt_1': rlrl1,
                                                  'leftsided_pe': lspe,
                                                  'chronic_pe': cpe,
                                                  'rightsided_pe': rspe,
                                                  'acute_and_chronic_pe': aacpe,
                                                  'central_pe': cnpe,
                                                  'indeterminate': indt})

        self._compile_model()

        return self.model

    def _compile_model(self):
        opt = Adam(lr=0.001)
        self.model.compile(loss="binary_crossentropy",
                           optimizer=opt, metrics=["accuracy"])

    def save_model_as_json(self):
        model_json = self.model.to_json()
        with open(os.path.join(self.path_to_save, "model.json"), "w") as json_file:
            json_file.write(model_json)

    def load_model_from_json(self, json, path_to_weights):
        self.model = model_from_json(json)
        self.model.load_weights(path_to_weights)
        self._compile_model()

    @staticmethod
    def get_pre_process_function():
        return preprocess_input

    # def get_generators(self, df, x_col, y_col, do_aug, path=None):
    #     if do_aug:
    #         img_gen = ImageDataGenerator(preprocessing_function=self.get_pre_process_function(),
    #                                      rescale=1. / 255,
    #                                      zoom_range=0.2,
    #                                      horizontal_flip=True,
    #                                      vertical_flip=True,
    #                                      brightness_range=[0.2, 1.5])
    #     else:
    #         img_gen = ImageDataGenerator(preprocessing_function=self.get_pre_process_function())
    #     generator = img_gen.flow_from_dataframe(dataframe=df,
    #                                             directory=path if path is not None else self.path_to_image,
    #                                             x_col=x_col,
    #                                             y_col=y_col,
    #                                             batch_size=self.batch,
    #                                             seed=2020,
    #                                             shuffle=True,
    #                                             class_mode="other",
    #                                             color_mode="rgb",
    #                                             target_size=self.size)
    #     return generator
