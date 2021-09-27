from datasets import snukb
import tensorflow_datasets as tfds
import tensorflow as tf


def build_batch_pipeline(dataset_name,
                         split: str,
                         buffer_size: int,
                         batch_size: int,
                         functions_before_batch: list = [],
                         functions_after_batch: list = []):


    dataset = tfds.load(dataset_name, split=split, as_supervised=True)

    before_batch = dataset.cache()
    for f in functions_before_batch:
        before_batch = before_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    after_batch = before_batch.shuffle(buffer_size).batch(batch_size)
    for f in functions_after_batch:
        after_batch = after_batch.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    batchs = after_batch.prefetch(tf.data.AUTOTUNE)

    return (batchs)