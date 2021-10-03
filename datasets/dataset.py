from datasets import snukb
import tensorflow_datasets as tfds
import tensorflow as tf
import csv
from util import text_processing
from util import dataset_util
from tensorflow_text.python.ops.tokenization import Tokenizer
from pathlib import Path

def get_data_generator(path,tokenizer):
    def data_generator():
        label_path = path / f'{path.parts[-1]}.csv'
        max_length = dataset_util.get_max_len_label_tokens(label_path)
        with label_path.open() as f:
            for row in csv.DictReader(f):
                image_id = row['index']
                image_path = str(path / 'images' / f'{image_id}.jpg')
                label = tokenizer.tokenize(row['label'],max_length=max_length)
                label = tf.constant(label,dtype=tf.int16)
                yield (image_path, label)
    return data_generator

def build_batch_pipeline(# data_path: str,
                         # tokenizer: Tokenizer,
                         dataset_name: str,
                         train_percent: int,
                         buffer_size: int,
                         batch_size: int,
                         functions_before_batch: list = [],
                         functions_after_batch: list = []):

    train_dataset = tfds.load(dataset_name, split=f'train[:{train_percent}%]', as_supervised=True)
    valid_dataset = tfds.load(dataset_name, split=f'train[{train_percent}%:]', as_supervised=True)
    '''
    data_path = Path(data_path)
    dataset = tf.data.Dataset.from_generator(get_data_generator(data_path, tokenizer),
                                             output_types=(tf.string, tf.int16),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([None]))
                                             )
    '''
    train_before_batch = train_dataset.cache()
    for f in functions_before_batch:
        train_before_batch = train_before_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    train_after_batch = train_before_batch.shuffle(buffer_size).batch(batch_size)
    for f in functions_after_batch:
        train_after_batch = train_after_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    train_batchs = train_after_batch.prefetch(tf.data.AUTOTUNE)

    valid_before_batch = valid_dataset.cache()
    for f in functions_before_batch:
        valid_before_batch = valid_before_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    valid_after_batch = valid_before_batch.shuffle(buffer_size).batch(batch_size)
    for f in functions_after_batch:
        valid_after_batch = valid_after_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    valid_batchs = valid_after_batch.prefetch(tf.data.AUTOTUNE)

    return (train_batchs), (valid_batchs)

