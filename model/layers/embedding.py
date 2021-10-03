import tensorflow as tf

class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(ImageEmbedding, self).__init__()
        self.patch_size = [1, patch_size[0], patch_size[1], 1]
    def call(self,imgs):
        x = tf.image.extract_patches(imgs,
                                     sizes=self.patch_size,
                                     strides=self.patch_size,
                                     rates=[1,1,1,1],
                                     padding='VALID')
        x = tf.reshape(x, shape=(tf.shape(x)[0],tf.shape(x)[1] * tf.shape(x)[2],tf.shape(x)[3]))
        return x