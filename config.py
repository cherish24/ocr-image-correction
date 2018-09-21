
'''config'''


## TRAINING VARIABLES
img_h = 64
img_w = 256
filters = 16
kernel_size = (3, 3)
pool_size = 2
rnn_dense = 32
rnn_size = 512
vocab_size = 24
batch_size = 32
downsample_factor = 8
max_size = 15

## DATA GENERATION VARIABLES
data_file = "data/values.gz"
images = "images"
ocr_images = "ocr_images"
altos = "alto"
ocr_texts = "ocr_texts"
ocr_extra_px = 5
train_size = 0.75
