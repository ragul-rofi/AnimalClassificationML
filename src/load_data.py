import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  


def load_data(base_dir = 'new_dataset', target_size=(224,224), batch_size=32):
    #creating seperate Imagedatagenrators for training,validate and test

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale =1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    #loading data from processed dataset
    train_data = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size= target_size,
        batch_size = batch_size,
        class_mode = 'categorical'
    )

    validation_data = val_datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, validation_data, test_data
