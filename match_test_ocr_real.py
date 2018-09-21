import os, gzip, json, tqdm
import config

def get_words_per_file():
	wpf = {}
	v = json.load(gzip.open(config.data_file, "rt"))
	for line in v:
		doc_id = line[3]
		split = doc_id.split("_")
		num = split.pop(-1)
		for i in range(0, 3-len(num)):
			num = "0" + num
		split.append(num + ".xml")
		doc_id = "_".join(split)
		ocr = line[7]
		corr = line[5]
		wpf[doc_id] = wpf.get(doc_id, {})
		wpf[doc_id][ocr] = wpf[doc_id].get(ocr, set())
		wpf[doc_id][ocr].add(corr)
	return wpf

def match_ocr_and_real(wpf, dataset_type):
	for txt_file in tqdm.tqdm([f for f in os.listdir(config.ocr_texts + "/" + dataset_type) if "_real" not in f], desc="Matching {} files...".format(dataset_type)):
		word = open(config.ocr_texts + "/" + dataset_type + "/" + txt_file, "r").read()
		doc_id = "_".join(txt_file.split("_")[:-1]) + ".xml"
		match = wpf[doc_id]
		if word not in match:
			continue
	#	if len(match[word]) > 1:
#			print(match[word], word)
		open(config.ocr_texts + "/" + dataset_type + "/" + txt_file.replace(".txt", "_real.txt"), "w").write("\n".join(match[word]))


if __name__ == "__main__":

	for dataset_type in ["test", "train"]:
		words_per_file = get_words_per_file()
		match_ocr_and_real(words_per_file, dataset_type)
