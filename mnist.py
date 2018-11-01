
from lenet import LeNet5
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np



epochs = 20
weightsPath = "lenet_weights.hdf5"
TRAIN_FLAG = 1

def load_mnist():

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
		
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	return x_train, x_test, y_train, y_test


if __name__ == "__main__":

	print("[INFO] loading mnist dataset...")
	x_train, x_test, y_train, y_test = load_mnist()

	# initialize the optimizer and model
	print("[INFO] compiling model...")

	model = LeNet5.build()	


	# if no weights specified train the model
	if TRAIN_FLAG:
	
		print("[INFO] training...")
		model.fit(x_train, y_train, batch_size=128, epochs=epochs,
			verbose=1)

		# show the accuracy on the testing set
		print("[INFO] evaluating...")
		(loss, accuracy) = model.evaluate(x_test, y_test,
			batch_size=128, verbose=1)

		print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

		print("[INFO] dumping weights to file...")
		model.save_weights(weightsPath, overwrite=True)
	else:
		model.load_weights(weightsPath)

	
	x_pred = x_test[0:10]
	y_pred = model.predict(x_pred)
	labels = np.argmax(y_pred, axis = 1)

	print(np.argmax(y_test[0:10], axis = 1))

	fix, axes = plt.subplots(2, 5, figsize=(15,8),
				subplot_kw={"xticks":(), "yticks":()})

	for label, image, ax in zip(labels, x_pred, axes.ravel()):
		ax.imshow(image.reshape(28, 28))
		ax.set_title(label)
	plt.show()

