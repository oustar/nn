
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Flatten

class LeNet5:
    """ LeNet-5

        input : 28 * 28
        C1 : 6 feature maps, kernel size 5*5, activation relu
        S2 : 6 feature maps, pooling size 2*2, maxpooling
        C3 : 16 feature maps, kernel size 5*5, activation relu
        S4 : 16 feature maps, pooling size 2*2, maxpooling
        C5 : 120 feature maps, kernel size 5*5, activation relu
        F6 : 84
        output : 10

    """
    @staticmethod
    def build():

        model = Sequential()

        model.add(ZeroPadding2D(padding=(2, 2), input_shape = (28, 28, 1)))

        # C1
        model.add(Conv2D(filters = 6, kernel_size = (5, 5), 
            strides = (1, 1), activation = 'relu' ))

        # S2
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # C3
        # TODO:
        model.add(Conv2D(filters = 16, kernel_size = (5, 5),            
            strides = (1, 1), activation = 'relu'))

        # S4
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        # C5
        model.add(Conv2D(filters = 120, kernel_size = (5, 5),            
            strides = (1, 1), activation = 'relu'))

        model.add(Flatten())

        # F6
        model.add(Dense(units = 84, activation = 'relu'))

        # output
        model.add(Dense(units = 10, activation = 'softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        return model