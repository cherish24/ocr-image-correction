import os, zipfile, gzip, json, tqdm

def get_ids(id_loc):
	d = json.load(gzip.open(id_loc, "rt"))
	ids = []
	for line in d:
		doc_id = line[3]
		split = doc_id.split("_")
		num = split.pop(-1)
		for i in range(0, 3-len(num)):
			num = "0" + num
		split.append(num + ".xml")
		doc_id = "_".join(split)
		ids.append(doc_id)
	return set(ids)

def get_altos(ids, zip_loc):
	altos = {}
	for zipf in tqdm.tqdm(os.listdir(zip_loc)):
		if not zipf.endswith(".zip"):
			continue
		with zipfile.ZipFile(zip_loc + "/" + zipf) as zip_file:
			for member in zip_file.namelist():
				if not member.endswith(".xml"):
					continue
				doc_id = member.split("/")[3]
				if doc_id in ids:
					print("FOUND ONE: {}".format(doc_id))
					altos[doc_id] = zip_file.open(member).read().decode()
	return altos


def save_altos(altos, save_loc):
	for alto_key, alto_data in altos.items():
		open(save_loc + "/" + alto_key, "w").write(alto_data)

if __name__ == "__main__":

	id_loc = "data/values.gz"
	zip_loc = "/home/aeniva/data/newspapers"
	zip_loc2 = "/home/aeniva/data/journals"
	save_loc = "alto/test"
	ids = get_ids(id_loc)
	print(ids)
	print(len(ids))
	altos1 = get_altos(ids, zip_loc)
	altos2 = get_altos(ids, zip_loc2)
	altos = {}
	altos.update(altos1)
	altos.update(altos2)
	save_altos(altos, save_loc)
