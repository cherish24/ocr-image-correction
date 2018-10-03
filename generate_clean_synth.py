import argparse, gzip, json, tqdm
from PIL import Image, ImageFont, ImageDraw

class SynthGenerator:

	def __init__(self, data_loc, save_loc, num_images, max_word_length, font_size, font):
		self.data_loc = data_loc
		self.save_loc = save_loc
		self.max_word_length = max_word_length
		self.font_size = font_size
		self.num_images = num_images
		self.font = font

	def load_font(self):
		font = ImageFont.truetype(self.font, self.font_size)
		return font

	def get_word_generator(self):
		while True:
			with gzip.open(self.data_loc, "rt") as data_file:
				for word in data_file:
					word = word.strip()
					if len(word) < 2 or len(word) > self.max_word_length:
						continue
					yield word

	def synthesize_images(self):
		font = self.load_font()
		word_generator = self.get_word_generator()
		for i in tqdm.tqdm(range(0, self.num_images), desc="Generating clean images..."):
			word = next(word_generator)
			image = Image.new("L", size=(2000, 2000), color="white")
			draw = ImageDraw.Draw(image)
			img_width, img_height = draw.textsize(word, font=font)
			draw.text((0, 0), word, font=font)
			image = image.crop((0, 10, img_width + 5, img_height + 5))
			s = image.size
		#	if s[0] > 500:
			#	image.show()
				#import time
				#time.sleep(7)
		#	print(word)
			image.show()
		#	image.save(self.save_loc + "/" + str(i) + ".png")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generating word level images of clean fraktur texts")
	parser.add_argument("--data-location", help="File with words to generate pictures from.", required=True)
	parser.add_argument("--save-location", help="Where to save the generated images.", required=True)
	parser.add_argument("--num-images", help="Number of images to generate.", default=10000, type=int)
	parser.add_argument("--max-word-length", help="Maximum word length to generate.", default=7)
	parser.add_argument("--font-size", help="Font size to use.", default=15, type=int)
	parser.add_argument("--font", help="What font to use.", default="Fraktur")

	args = parser.parse_args()

	gen = SynthGenerator(data_loc=args.data_location, save_loc=args.save_location, num_images=args.num_images, max_word_length=args.max_word_length, font_size=args.font_size, font=args.font)
	gen.synthesize_images()
