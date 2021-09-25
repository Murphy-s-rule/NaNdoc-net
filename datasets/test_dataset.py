import unittest
from util import dataset_util
from tensorflow_text import UnicodeCharTokenizer
from datasets import dataset

class Dataset(unittest.TestCase):

    def test_build_batch_pipeline(self):
        tokenizer = UnicodeCharTokenizer()

        BUFFER_SIZE = 2000
        BATCH_SIZE = 64
        WIDTH = 380
        HEIGHT = 380
        CHANNEL = 3
        MAX_SEQ_LENGTH = 50

        train_batches = dataset.build_batch_pipeline('snukb', 'train',
                                                buffer_size=BUFFER_SIZE,
                                                batch_size=BATCH_SIZE,
                                                functions_before_batch=[
                                                    dataset_util.get_resize_image_func(WIDTH,HEIGHT,
                                                                                       is_normalize_pixel=True,
                                                                                       normalization_value=255)],
                                                functions_after_batch=[
                                                    dataset_util.get_tokenize_label_func(tokenizer,
                                                                                         is_sequence_padding=True,
                                                                                         max_seq_length=MAX_SEQ_LENGTH)]
                                                )
        for images, labels in train_batches.take(1):
            self.assertEqual(images.shape, (BATCH_SIZE,WIDTH,HEIGHT,CHANNEL),
                             f'image shape : {images.shape} \t testsheet : {(BATCH_SIZE,WIDTH,HEIGHT,CHANNEL)}')
            self.assertEqual(labels[0].shape, (64,MAX_SEQ_LENGTH),
                             f'sequence shape : {labels[0].shape} \t testsheet : {(64,MAX_SEQ_LENGTH)}')
            self.assertEqual(labels[1].shape, (64, MAX_SEQ_LENGTH),
                             f'sequence shape : {labels[1].shape} \t testsheet : {(64, MAX_SEQ_LENGTH)}')


        test_batches = dataset.build_batch_pipeline('snukb', 'test',
                                                buffer_size=BUFFER_SIZE,
                                                batch_size=BATCH_SIZE,
                                                functions_before_batch=[
                                                    dataset_util.get_normalize_pixel_func(normalization_value=255),
                                                    dataset_util.get_resize_image_func(WIDTH, HEIGHT,
                                                                                       is_normalize_pixel=False)],
                                                functions_after_batch=[
                                                    dataset_util.get_tokenize_label_func(tokenizer,
                                                                                         is_sequence_padding=True,
                                                                                         max_seq_length=MAX_SEQ_LENGTH)]
                                                )

        for images, labels in test_batches.take(1):
            self.assertEqual(images.shape, (BATCH_SIZE,WIDTH,HEIGHT,CHANNEL),
                             f'image shape : {images.shape} \t testsheet : {(BATCH_SIZE,WIDTH,HEIGHT,CHANNEL)}')
            self.assertEqual(labels[0].shape, (64,MAX_SEQ_LENGTH),
                             f'sequence shape : {labels[0].shape} \t testsheet : {(64,MAX_SEQ_LENGTH)}')
            self.assertEqual(labels[1].shape, (64, MAX_SEQ_LENGTH),
                             f'sequence shape : {labels[1].shape} \t testsheet : {(64, MAX_SEQ_LENGTH)}')

if __name__ == '__main__':
    unittest.main()