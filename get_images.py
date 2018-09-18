import os
import config
import time

from logger import get_logger

''' Download test and train images from Kansalliskirjasto '''


def get_urls(data_type):
	urls = {}
	for alto_file in os.listdir(config.altos + "/" + data_type):
		alto = open(config.altos + "/" + data_type + "/" + alto_file).read()
		url = alto.split("<imageURL>")[1].split("</imageURL>")[0]
		urls[alto_file] = url
	return urls

def download_files(urls, data_type):
	for file_name, url in urls.items():
		time.sleep(1)
		os.system("wget {} -O {}".format(url, config.images + "/" + data_type + "/" + file_name.replace(".xml", ".png")))

if __name__ == "__main__":

	logger = get_logger()
	for data_type in ["test", "train"]:
			logger.info("Getting {} images...".format(data_type))
			urls = get_urls(data_type)
			download_files(urls, data_type)
