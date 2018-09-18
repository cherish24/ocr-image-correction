import os, tqdm

import config
from PIL import Image
from logger import get_logger
from lxml import etree as ET

''' Extract sub images of words and the OCR'd text from the full document '''

def extract_ocr_images(alto, image):
	ocr_images = []
	for event, elem in ET.iterparse(alto, events=("start", "end")):
		if event == "start":
			if elem.tag == "{kk-ocr}Page":
				ocr_page_height = elem.get("HEIGHT")
				ocr_page_width = elem.get("WIDTH")
				hm = int(ocr_page_height) / image.size[1]
				wm = int(ocr_page_width) / image.size[0]
				#print("OCR HEIGHT:{}\tOCR WIDTH:{}\t ACTUAL HEIGHT:{}\t ACTUAL WIDTH:{} \tHM:{}\tWM:{}".format(ocr_page_height, ocr_page_width, image.size[1], image.size[0], hm, wm))
		elif event == "end":
			if elem.tag == "{kk-ocr}TextLine":
				images = []
				words = []
				for word_element in elem.iter("{kk-ocr}String"):
					word = word_element.get("CONTENT")
					horizontal_pos = int(word_element.get("HPOS"))
					vertical_pos = int(word_element.get("VPOS"))
					height = int(word_element.get("HEIGHT"))
					width = int(word_element.get("WIDTH"))
					s1 = int(horizontal_pos/hm)-config.ocr_extra_px
					s2 = int(vertical_pos/wm)-config.ocr_extra_px
					e1 = int((horizontal_pos+width)/hm) + config.ocr_extra_px
					e2 = int((vertical_pos+height)/wm) + config.ocr_extra_px
					sub_image = image.crop((s1,s2,e1,e2))
					images.append(sub_image)
					words.append(word)
				ocr_images.append([" ".join(words), images])
	return ocr_images

def extract(data_type):
	for alto_file in tqdm.tqdm(os.listdir(config.altos + "/" + data_type), desc="Extracting OCR sub-images..."):
		alto = open(config.altos + "/" + data_type + "/" + alto_file, "rb")
		image = Image.open(config.images + "/" + data_type + "/" + alto_file.replace(".xml", ".png"))
		ocr_images = extract_ocr_images(alto, image)
		for ocr_image_i, ocr_image_data in enumerate(ocr_images):
			ocr_text, ocr_word_images = ocr_image_data
			for ocr_word_image_i, ocr_word_image in enumerate(ocr_word_images):
				ocr_word_image.save(config.ocr_images + "/" + data_type + "/" + alto_file.split(".xml")[0] + "_" + str(ocr_image_i) + "_" + str(ocr_word_image_i) + ".png")
			open(config.ocr_texts + "/" + data_type + "/" + alto_file.split(".xml")[0] + "_" + str(ocr_image_i) + ".txt", "w").write(ocr_text)

if __name__ == "__main__":

	logger = get_logger()
	for data_type in ["train", "test"]:
		logger.info("Extracting OCR sub-images from {} set".format(data_type))
		extract(data_type)
