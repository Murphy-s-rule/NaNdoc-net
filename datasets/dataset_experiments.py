import csv
import tensorflow as tf
from util.text_processing import HangulTokenizer
from pathlib import Path
from tensorflow_text.python.ops.tokenization import Tokenizer
from util import dataset_util
from datasets.dataset import build_batch_pipeline


image_size = (380,380)
tokenizer = HangulTokenizer()
BUFFER_SIZE = 2000
BATCH_SIZE = 32

train_path = 'datasets/snukb/dataset/train'
train_batches = build_batch_pipeline(data_path=train_path,
                                     tokenizer=tokenizer,
                                     buffer_size=BUFFER_SIZE,
                                     batch_size=BATCH_SIZE)

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    return img

for (batch, (image_paths, tar)) in enumerate(train_batches):

    images = [load_image(image_path) for image_path in image_paths.numpy()]
    images = tf.concat(images,axis=0)
    print(f'images : {images}')