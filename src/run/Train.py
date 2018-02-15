from keras.datasets import mnist
import numpy as np
import cv2
from keras.models import Model, Input
import matplotlib.pyplot as plt
from model import EncDecModel as model

from model import DeepFakeModel as deepmodel
import os

img_shape = (64,64,3)
#(x_train, _), (x_test, _) = mnist.load_data()


inp_img = []
directory = os.fsencode('C:/Users/David/PycharmProjects/DeepFake/src/image')
for file in os.listdir(directory):
    pass
    try:

        filepath = os.fsdecode(os.path.join(directory, file))
        x_train = cv2.imread(filepath)

        x_train = cv2.resize(x_train, (64, 64))

        x_train = x_train.astype('float32') / 255.

        #x_train = cv2.cvtColor(x_train, cv2.COLOR_BGR2GRAY)

        inp_img.append(x_train)

    except:
        pass
inp_img = np.array(inp_img)

(x_train, _), (x_test, _) = mnist.load_data()
#x_train = np.reshape(x_train, (1, 100, 100, 3))  # adapt this if using `channels_first` image data format


input_img = Input(shape=img_shape)

encoder = deepmodel.Encoder(nc_in=3,input_size=64)
decoder_A = deepmodel.Decoder_ps()
decoder_B = deepmodel.Decoder_ps()

m = Model(input_img, decoder_A(encoder(input_img)))

m.compile(optimizer='adadelta', loss='binary_crossentropy')


#inp_img = np.array([x_train])

x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
#inp_img = x_train[:1]
m.fit(inp_img, inp_img, epochs=500, batch_size=10, shuffle=True)

#m = Model(input_img,dec(enc(input_img)))
#m.compile(optimizer='adadelta', loss='binary_crossentropy')

out = m.predict(inp_img)
n = 10


#ax = plt.subplot(2, n, i)
for i in range(10):
    cv2.imshow("input" + str(i), inp_img[i])
    cv2.imshow("output" + str(i),out[i])
#cv2.imshow("hecvec",out[0])
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)


#cv2.imshow(out[i].reshape(100, 100))

cv2.waitKey()