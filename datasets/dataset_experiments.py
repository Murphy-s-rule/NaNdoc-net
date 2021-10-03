import csv
import tensorflow as tf
from util.text_processing import HangulTokenizer
from pathlib import Path
from tensorflow_text.python.ops.tokenization import Tokenizer
from util import dataset_util
from datasets.dataset import build_batch_pipeline
from datasets.snukb import snukb

image_size = (256,256)
patch_size = (8,8)
tokenizer = HangulTokenizer()
BUFFER_SIZE = 2000
BATCH_SIZE = 32

train_batches = build_batch_pipeline(dataset_name='snukb',
                                     buffer_size=BUFFER_SIZE,
                                     batch_size=BATCH_SIZE,
                                     functions_before_batch=[
                                         dataset_util.get_resize_image_func(image_size,
                                                                            is_normalize_pixel=True,
                                                                            normalization_value=255)
                                     ])

for (batch, (img, tar)) in enumerate(train_batches):
    print(img)
    print(tar)
    assert False