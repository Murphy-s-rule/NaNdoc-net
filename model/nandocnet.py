import tensorflow as tf

from model.layers.encoder import Encoder
from model.layers.decoder import Decoder

class NaNdocNet(tf.keras.Model):
    def __init__(self, patch_size, num_layers, d_model, num_heads, dff,
                 target_char_size, pe_input, pe_target, rate=0.1):
        super(NaNdocNet, self).__init__()



        self.encoder = Encoder(patch_size, num_layers, d_model, num_heads, dff, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_char_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_char_size)

    def call(self, img_embedded, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(img_embedded, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights