import tensorflow as tf

class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size,d_model):
        super(ImageEmbedding, self).__init__()
        self.patch_size = [1, patch_size[0], patch_size[1], 1]
        self.d_model = d_model
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self,imgs):
        x = tf.image.extract_patches(imgs,
                                     sizes=self.patch_size,
                                     strides=self.patch_size,
                                     rates=[1,1,1,1],
                                     padding='VALID')
        x = tf.reshape(x, shape=(tf.shape(x)[0],x.shape[1] * x.shape[2],x.shape[3]))
        x = self.dense(x)
        return x