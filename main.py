import glob
import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from my_utils import split_data, create_generators, skin_disease_model,predict_with_model

if __name__ == "__main__":

    path_to_images = "C:\\Users\\pknad\\OneDrive\\Documents\\Machine_Learning\\images\\"
    path_to_save_test = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Skin_Disease_CNN\Test"
    path_to_save_val = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Skin_Disease_CNN\\val"
    path_to_save_train = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Skin_Disease_CNN\Train"
    path_to_dataframe = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\GroundTruth.csv"
    path_to_save_model = "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Skin_Disease_CNN\Saved_Model"

    images_paths = glob.glob(os.path.join(path_to_images, "*.jpg"))
    df = pd.read_csv(path_to_dataframe)

    batch_size = 64
    epochs = 20

    train_generator, val_generator, test_generator = create_generators(batch_size=batch_size, train_data_path=path_to_save_train, val_data_path=path_to_save_val, test_data_path=path_to_save_test)

    num_classes = train_generator.num_classes
    TRAIN = False
    TEST = False
    EVAL = True

    if False:
        split_data(path_to_images=images_paths, path_to_save_test=path_to_save_test, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val, dataframe=df, image_path=path_to_images)

    if TRAIN:
        chpkt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_loss",
            mode="min",
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1,
        )

        early_stop = EarlyStopping(
            monitor = "val_accuracy",
            patience = 10,
        )

        model = skin_disease_model(num_classes=num_classes)

        model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])

        model.fit(train_generator, 
                batch_size= batch_size, 
                epochs=epochs, 
                validation_data= val_generator,
                callbacks = [chpkt_saver, early_stop]
                )
    
    if TEST:
        model = tf.keras.models.load_model(path_to_save_model)
        model.summary()

        print("Evaluating validation set: ")
        model.evaluate(val_generator)

        print("Evaluating test set")
        model.evaluate(test_generator)
    
    if EVAL:
        model = tf.keras.models.load_model(path_to_save_model)
        img_path= "C:\\Users\pknad\OneDrive\Documents\Machine_Learning\Skin_Disease_CNN\custom_data\DermatoFibra.JPG"
        predict_with_model(model,img_path=img_path)
