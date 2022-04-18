from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, UpSampling2D
from keras.layers import LeakyReLU
import numpy as np
from matplotlib import pyplot as plt

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    # encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(8, (3, 3), padding='same', input_shape=img_shape))
    encoder.add(LeakyReLU())
    encoder.add(Conv2D(32, (3, 3), padding='same'))
    encoder.add(LeakyReLU())
    encoder.add(MaxPooling2D(pool_size=(2, 2)))
    encoder.add(Conv2D(32, (3, 3), padding='same'))
    encoder.add(LeakyReLU())
    encoder.add(Conv2D(64, (3, 3), padding='same'))
    encoder.add(LeakyReLU())
    encoder.add(MaxPooling2D(pool_size=(2, 2)))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(12288))
    decoder.add(Reshape((12, 16, 64)))
    decoder.add(Conv2D(64, (3, 3), padding='same'))
    decoder.add(LeakyReLU())
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(32, (3, 3), padding='same'))
    decoder.add(LeakyReLU())
    decoder.add(Conv2D(32, (3, 3), padding='same'))
    decoder.add(LeakyReLU())
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2D(8, (3, 3), padding='same'))
    decoder.add(LeakyReLU())
    decoder.add(Conv2D(3, (3, 3), padding='same'))
    decoder.add(LeakyReLU())


    return encoder, decoder

def show_image(x):
    # plt.imshow(np.clip(x + 1.0, 0, 2))
    # plt.imshow(x*0.5+0.5)
    plt.imshow(x)

def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,2,1)
    plt.title("Original")
    show_image(img)

    # plt.subplot(1,3,2)
    # plt.title("Code")
    # plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()