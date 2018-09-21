import os, sys, tqdm
import config
from random import shuffle

if __name__ == "__main__":

	alto_files = os.listdir(config.altos + "/test")
	shuffle(alto_files)
	train_files = alto_files[:int(len(alto_files)*config.train_size)]
	for train_file in tqdm.tqdm(train_files, desc="Splitting test set into train..."):
		image_file = train_file.replace(".xml", ".png")
		os.system("mv {} {}".format(config.altos + "/test/" + train_file, config.altos + "/train/" + train_file))
		os.system("mv {} {}".format(config.images + "/test/" + image_file, config.images + "/train/" + image_file))
