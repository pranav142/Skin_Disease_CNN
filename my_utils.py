
from sklearn.model_selection import train_test_split
import os
import shutil
import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def split_data(path_to_images, path_to_save_test, path_to_save_val, path_to_save_train,  dataframe, image_path, split_size=0.1):

    Disease_Mapping = {
        'MEL\n': 0,
        'NV\n' : 1,
        'BCC\n': 2,
        'AKIEC\n': 3,
        'BKL\n': 4,
        'DF\n': 5,
        'VASC\n': 6,
    }          
    train_init, test_data = train_test_split(path_to_images, test_size=split_size)
    train_data, val_data = train_test_split(train_init, test_size=split_size)

    
    for image in train_data:

        image_name = image.replace(image_path, '')
        image_name = image_name.replace('.jpg','')
        print(image_name)
        row = dataframe.loc[(dataframe["image"] == image_name)]
        row = row.select_dtypes(include=['float64', 'int64'])
        row = row.reset_index(drop = True)
    
        column_index = row.idxmax(axis = 1)
        column_index = str(column_index).replace("0    ",'')
        column_index = column_index.replace("dtype: object",'')
        folder = str(Disease_Mapping[column_index])

        path_to_folder = os.path.join(path_to_save_train, folder)
        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)

        shutil.copy(image, path_to_folder)
    

    for image in val_data:

        image_name = image.replace(image_path, '')
        image_name = image_name.replace('.jpg','')
        row = dataframe.loc[(dataframe["image"] == image_name)]
        row = row.select_dtypes(include=['float64', 'int64'])
        row = row.reset_index(drop = True)

        column_index = row.idxmax(axis = 1)
        column_index = str(column_index).replace("0    ",'')
        column_index = column_index.replace("dtype: object",'')
        folder = str(Disease_Mapping[column_index])

        path_to_folder = os.path.join(path_to_save_val, folder)
        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)

        shutil.copy(image, path_to_folder)

    for image in test_data:

        image_name = image.replace(image_path, '')
        image_name = image_name.replace('.jpg','')
        row = dataframe.loc[(dataframe["image"] == image_name)]
        row = row.select_dtypes(include=['float64', 'int64'])
        row = row.reset_index(drop = True)

        column_index = row.idxmax(axis = 1)
        column_index = str(column_index).replace("0    ",'')
        column_index = column_index.replace("dtype: object",'')
        folder = str(Disease_Mapping[column_index])

        path_to_folder = os.path.join(path_to_save_test, folder)
        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)

        shutil.copy(image, path_to_folder)


def skin_disease_model(num_classes):

    my_input = Input(shape=(100,100,3))

    x = Conv2D(32, (3,3), activation = 'relu')(my_input)
    x = Conv2D(64, (3,3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation = 'relu')(x)
    x = Conv2D(256, (3,3), activation = 'relu')(x)
    x = Conv2D(512, (3,3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model

def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    # preprocess images and scales pixel values by factor of 255
    preprocessor = ImageDataGenerator(
        rescale=1/255,
    )

    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode= "categorical",
        target_size= (100,100),
        color_mode= 'rgb',
        shuffle= 'True',
        batch_size = batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode= "categorical",
        target_size= (100,100),
        color_mode= 'rgb',
        shuffle= 'True',
        batch_size = batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode= "categorical",
        target_size= (100,100),
        color_mode= 'rgb',
        shuffle= 'False',
        batch_size = batch_size
    )

    return train_generator, val_generator, test_generator

def print_img(img):
    plt.figure(figsize=(10,10))
    plt.tight_layout()
    plt.imshow(img)
    plt.show()

def predict_with_model(model, img_path):

    Disease_Mapping = {
        0: "melanoma",
        1: "melanocytic nevi",
        2: "basal cell carcinoma",
        3: "Actinic keratoses",
        4: "benign keratosis",
        5: "dermatofibroma",
        6: "vascular lesions", 
    }          

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels = 3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [100,100])
    print_img(img)
    img = tf.expand_dims(img, axis=0) 

    prediction = model.predict(img)
    print(prediction)
    prediction = np.argmax(prediction)

    prediction = Disease_Mapping[prediction]
    print(f"Prediction = {prediction}")
    return prediction
