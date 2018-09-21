import keras
import tensorflow as tf
import allow_growth
import config

import os, json, random
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.layers import Input, Dense, Activation, Reshape, Lambda, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
#from PIL import Image
import cv2
import numpy as np
import itertools

sess = tf.Session()
K.set_session(sess)

def ctc_loss_function(args):
	y_pred, labels, input_length, label_length = args
	##TODO ADD WEIRD HACK?
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(vocab):

	## img model
	inp = Input(name="image_input", shape=(config.img_w, config.img_h, 1), dtype="float32")
	#inp = Input(name="image_input", shape=(1, config.img_w, config.img_h), dtype="float32")
	c = Conv2D(config.filters, config.kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="Conv_1")(inp)
	c = MaxPooling2D(pool_size=(config.pool_size, config.pool_size))(c)
	c = Conv2D(config.filters, config.kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="Conv_2")(c)
#	c = GlobalMaxPooling2D()(c)
	c = MaxPooling2D(pool_size=(config.pool_size, config.pool_size))(c)

	conv_to_rnn = (config.img_w // (config.pool_size ** 2), (config.img_h // (config.pool_size ** 2)) * config.filters)
	c = Reshape(target_shape=conv_to_rnn, name="cnn_to_rnn_reshape")(c)

	## 'downsample'
	c = Dense(config.rnn_dense, activation="relu", name="RNN_dense")(c)

	gru_1 = GRU(config.rnn_size, return_sequences=True, kernel_initializer="he_normal", name="GRU1")(c)
	gru_1_back = GRU(config.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name="GRU1_back")(c)
	gru_1_merge = add([gru_1, gru_1_back])
	gru_2 = GRU(config.rnn_size, return_sequences=True, kernel_initializer="he_normal", name="GRU2")(gru_1_merge)
	gru_2_back = GRU(config.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name="GRU2_back")(gru_1_merge)
	concatenated_gru_2 = concatenate([gru_2, gru_2_back])
	## RNN to character activations

	c = Dense(len(vocab)+1, kernel_initializer="he_normal", name="Character_activation_dense")(concatenated_gru_2)
	y_pred = Activation("softmax", name="activations")(c)

	img_model = Model(inputs=[inp], outputs=[y_pred])
	img_model.summary()

	labels = Input(name="labels", shape=[config.max_size], dtype="float32")
	input_length = Input(name="input_length", shape=[1], dtype="int64")
	label_length = Input(name="label_length", shape=[1], dtype="int64")
	ctc_loss = Lambda(ctc_loss_function, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

#	optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	optimizer = Adam()
	model = Model(inputs=[inp, labels, input_length, label_length], outputs=[ctc_loss])
	model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

	return model

def pad_image(img):
	##TODO PAD IMAGE TO 256
	pass


def image_to_matrix(image):
	#shape = image.shape
#	new_image = cv2.copyMakeBorder(image, 0, config.img_h-shape[0], 0, config.img_w-shape[1], cv2.BORDER_CONSTANT,value=[255,255,255])
	#cv2.imshow("image", new_image)
#	cv2.waitKey(0)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (config.img_w, config.img_h))
	image = image.astype(np.float32)
	image /= 255
	# width and height are backwards from typical Keras convention
	# because width is the time dimension when it gets fed into the RNN
	return image

def text_to_vector(text, vocab):
	text = text[:config.max_size]
	v = np.zeros(config.max_size, dtype="int32")
	for char_i, char in enumerate(text):
		v[char_i] = vocab.get(char, 1)
	return v

def read_image_and_text(images, i, dataset_type, vocab):
	#image = Image.open(config.ocr_images + "/" + dataset_type + "/" + images[i])
	image = cv2.imread(config.ocr_images + "/" + dataset_type + "/" + images[i])
	gold = open(config.ocr_texts + "/" + dataset_type + "/" + images[i].replace(".png", "_real.txt")).read().split("\n")[0] ## FIRST JUST NOW
	#ocr = open(config.ocr_texts + "/" + dataset_type + "/" + ocr_image.replace(".png", ".txt")).read()
	return image_to_matrix(image), text_to_vector(gold, vocab), gold

def get_image_text_generator(dataset_type, vocab, steps):
	images = sorted(os.listdir(config.ocr_images + "/" + dataset_type))
	d = 0
	while True:
		for i in range(0, len(images)):
			if not os.path.exists(config.ocr_texts + "/" + dataset_type + "/" + images[i].replace(".png", "_real.txt")):
				continue
			yield read_image_and_text(images, i, dataset_type, vocab)
			d += 1
			if d == steps:
				d = 0
				break

def get_data(dataset_type, vocab, steps=-1):
	#images = os.listdir(config.ocr_images + "/" + dataset_type)
	image_text_gen = get_image_text_generator(dataset_type, vocab, steps)
	X = []
	Y = []
	while True:
		x = np.ones([config.batch_size, config.img_w, config.img_h, 1])
		y = np.ones([config.batch_size, config.max_size])
		input_length = np.ones((config.batch_size, 1)) * (config.img_w // config.downsample_factor - 2) # wat
		label_length = np.zeros((config.batch_size, 1))
		for j in range(0, config.batch_size):
			image, text, r_text = next(image_text_gen)
			#print(r_text)
		#	cv2.imshow("image", image)
		#	cv2.waitKey(0)
			image = image.T
			image = np.expand_dims(image, -1)
			x[j] = image
			y[j] = text
			label_length[j] = len(text)
		inputs = {"image_input": x, "labels": y, "input_length": input_length, "label_length": label_length}
		output = {"ctc": np.zeros([config.batch_size])}

		yield (inputs, output)
		#X.append(inputs)
		#Y.append(output)

	#return X, Y
def read_vocab():
	vocab = {"<MASK>": 0, "<OOV>": 1}
	for filename in os.listdir(config.ocr_texts + "/train/"):
		if "real" not in filename: continue
		text = open(config.ocr_texts + "/train/" + filename, "r").read()
		for char in text:
			if char not in vocab:
				vocab[char] = len(vocab)
	return vocab

def decode_batch(out, vocab):
	preds = []
	for i in range(out.shape[0]):
		best = list(np.argmax(out[i, :], 1))
		out_v = [k for k, g in itertools.groupby(best)]
		text = []
		for char_i in out_v:
			if char_i in vocab:
				text.append(vocab[char_i])
		preds.append("".join(text))
	return preds

def remove_masks(text):
	return text.replace("<MASK>", "")


def decode(val_gen, model, vocab):
	input_layer = model.get_layer(name="image_input").input
	output_layer = model.get_layer(name="activations").output
	rev_vocab = {v:k for k,v in vocab.items()}
	for i in range(0, 1000//config.batch_size): # batches
		inputs, output = next(val_gen)
		X = inputs["image_input"]
		network_out = sess.run(output_layer, feed_dict={input_layer:X})
		predicted_texts = decode_batch(network_out, rev_vocab)
		labels = inputs["labels"]
		texts = []
		for label in labels:
			text = "".join([rev_vocab[c] for c in label])
			texts.append(text)
		for pred, real in zip(predicted_texts, texts):
			print("Pred:", remove_masks(pred))
			print("Real:", remove_masks(real))
			print()
		#print(predicted_texts)
	#	print(texts)

		#for i in range(0, config.batch_size):

if __name__ == "__main__":
	# train_X, train_y = get_data("train")
	vocab = read_vocab()
	train_gen = get_data("train", vocab, 10000)
	val_gen = get_data("test", vocab, 10000)
	# val_X, val_y = get_data("test")
	model = get_model(vocab)
#	model.fit_generator(generator=train_gen, steps_per_epoch=50000//config.batch_size//100, epochs=20, validation_data=val_gen, validation_steps=50000//config.batch_size//5//100)
	model.fit_generator(generator=train_gen, steps_per_epoch=10000//config.batch_size, epochs=50, validation_data=val_gen, validation_steps=10000//config.batch_size)
	#train_gen = get_data("train", vocab, 50)
	decode(val_gen, model, vocab)
	# next(train_gen)
	#model.train_on_batch(train_X[0], train_y[0])
	#model.train_on_batch(train_X[0], train_y[0], epochs=10, validation_data=(val_X[0], val_y[0]))
