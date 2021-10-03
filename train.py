import tensorflow as tf
import numpy as np
import time
import argparse

from datasets import dataset
from util import dataset_util
from util.text_processing import HangulTokenizer
from model.nandocnet import NaNdocNet
from util import model_util

def get_argparser():
    parser = argparse.ArgumentParser(description='Train NaNdocNet')
    parser.add_argument('--buffer-size', type=int, default=2000, required=False,
                        help='Buffer size used to crate mini batch')
    parser.add_argument('--batch-size', type=int, default=32, required=False,
                       help='Batch_size is amount data fair each mini batch')
    parser.add_argument('--epochs', type=int, default=50, required=False,
                        help='Epochs is count iteration to train dataset')
    parser.add_argument('--num-layers', type=int, default=4, required=False,
                        help='Count encoder and decoder layer')
    parser.add_argument('--num-heads', type=int, default=8, required=False,
                        help='Amount attention head')
    parser.add_argument('--d-model', type=int, default=256, required=False,
                        help='charactor embedding dimension')
    parser.add_argument('--dff', type=int, default=512, required=False,
                        help='dimension of hidden layer in point-wise network')
    parser.add_argument('--dropout-rate', type=float, default=0.1, required=False,
                        help='dropout rate when train')
    parser.add_argument('--image-resize', type=int, nargs="+", default=[128, 128], required=False,
                        help='image size resizing before training')
    parser.add_argument('--patch-size', type=int, nargs="+", default=[8, 8], required=False,
                        help='patch size when patch for embedding')
    parser.add_argument('--dataset-name', type=str, default='snukb', required=False,
                        help='custom dataset name')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/train/defualt', required=False,
                        help='checkpoint path to save trained model')
    return parser

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def train(buffer_size=2000, batch_size=32, epochs=50,
          image_size=(128,128), patch_size=(8,8),
          num_layers=4, d_model=256, dff=512, num_heads=8, dropout_rate=0.1,
          dataset_name='snukb', checkpoint_path='./checkpoints/train/defualt'):


    image_seq_len = int(image_size[0] / patch_size[0]) * int(image_size[1] / patch_size[1])

    tokenizer = HangulTokenizer()
    target_char_size = tokenizer.char_size

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    transformer = NaNdocNet(patch_size, num_layers, d_model, num_heads, dff, target_char_size,
                              pe_input=image_seq_len + 1,
                              pe_target=target_char_size,
                              rate=dropout_rate)

    train_step_signature = [
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, image_seq_len), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int16),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(img, img_mask_base, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = model_util.create_masks(img_mask_base, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(img, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_batches = dataset.build_batch_pipeline(dataset_name=dataset_name,
                                         buffer_size=buffer_size,
                                         batch_size=batch_size,
                                         functions_before_batch=[
                                             dataset_util.get_resize_image_func(image_size,
                                                                                is_normalize_pixel=True,
                                                                                normalization_value=255)
                                         ])

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (img, tar)) in enumerate(train_batches):
            img_mask_base = np.ones(shape=(img.shape[0], image_seq_len))
            train_step(img, img_mask_base, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    image_size = tuple(args.image_resize)
    patch_size = tuple(args.patch_size)
    d_model = patch_size[0] * patch_size[1] * 3

    train(buffer_size=args.buffer_size, batch_size=args.batch_size, epochs = args.epochs,
          image_size = image_size, patch_size = patch_size, num_layers = args.num_layers,
          d_model = d_model, dff = args.dff, num_heads = args.num_heads,
          dropout_rate = args.dropout_rate, dataset_name = args.dataset_name, checkpoint_path = args.checkpoint_path)