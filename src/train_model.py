from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from build_model import build_model
from load_data import load_data

def train_model():

    #Loading data
    train_data, validation_data ,_ = load_data()

    #Gettin number of classes from the training data

    num_classes = train_data.num_classes

    # Model Building

    model = build_model(num_class=num_classes)

    #Set up checkpoints and early stopping

    checkpoint = ModelCheckpoint('best_model.keras', monitor= 'val_loss',save_best_only = True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    #Training the model

    history = model.fit(
        train_data,
        validation_data = validation_data,
        epochs = 10,
        batch_size = 32,
        callbacks = [checkpoint,early_stop]
    )

    return model, history
        