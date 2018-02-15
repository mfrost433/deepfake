from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU
from keras.models import Model
from keras import backend as K
import tensorflow as tf
  # adapt this if using `channels_first` image data format
input_img = Input(shape=(28, 28, 1))
def load_model():
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoded, decoded

def Encoder():
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    model = Model(input_img, encoded)

    return model
def Decoder(inp = Input(shape=(5, 5, 1))):
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(inp)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    model = Model(inp, decoded)
    print(model.summary())
    return model


def load_model_2():
    model = Model(input_img,Decoder(Encoder(input_img)))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model