from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

def build_model(input_shape=(224,224,3),num_class = 5, use_transfer_learning = False):
    model = Sequential()

    if use_transfer_learning:
    #Using pre-trained model - MobileNetV2 but without top layer
        base_model = MobileNetV2(weight = 'imagenet', include_top=False, input_shape = input_shape)
        base_model.trainable = False
        model.add(base_model)
        model.add(GlobalAveragePooling2D)
    else:
        #Building model from scratch
        
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128,(3,3),activation= 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2)) #This prevents the overfitting
    model.add(Dense(num_class, activation='softmax')) # outputlayer

    model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model