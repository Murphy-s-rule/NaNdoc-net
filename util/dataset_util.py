import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as tftx

def get_resize_image_func(height,width,is_normalize_pixel,normalization_value=None):
    """
    The get_resize_image_func returns the closer function which resize image tensor in dataset pipeline.

    Args:
        height (int): Height of resized image
        width (int): Width of resized image
        is_normalize_pixel (bool): If true then processing normalize image before resizing image.
        normalization_value (int): Normalizing value when is_normalize_pixel is true.
    Returns:
        resize_image (function): The function resize image tensor and return resized and label
    """
    assert (not is_normalize_pixel) or normalization_value, \
        'cannot return resize image function. because is_normalize_pixel is true but normalization_value is None'
    @tf.function
    def resize_image(image, label):
        if is_normalize_pixel:
            normalize_pixel = get_normalize_pixel_func(normalization_value)
            image, _ = normalize_pixel(image, None)
        # Resize the image
        image = tf.image.resize(image, (height,width))

        return image, label

    return resize_image

def get_normalize_pixel_func(normalization_value):
    """
    The get_normalize_pixel_func returns the closer function which normalize pixel of image tensor in dataset pipeline.

    Args:
        normalization_value (int): Normalizing value when is_normalize_pixel is true.

    Returns:
        normalize_pixel (function): The function normalize pixel value between 0~1
    """
    @tf.function
    def normalize_pixel(image, label):
        image = tf.cast(image, tf.float32)
        # Normalize the pixel values
        image = image / normalization_value

        return image, label

    return normalize_pixel

def get_tokenize_label_func(tokenizer: tftx.Tokenizer,is_sequence_padding: bool,max_seq_length: int=None):
    """
    The get_tokenize_label_func returns the closer function which tokenize label text sequence tensor in dataset pipeline.

    Args:
        tokenizer (tftx.Tokenizer): tokenizer tokenize label text sequence to numbers.
        is_sequence_padding (bool): if true then append padding until label length is max_seq_length for equalization to processing parallel
        max_seq_length (int): max sequence length when is_sequence_padding is true

    Returns:
        tokenize_label (function): The function tokenize label to numbers.
    """
    assert (not is_sequence_padding) or max_seq_length, \
        'cannot return tokenize label function. because is_sequence_padding is true but max_seq_length is None'
    def tokenize_label(image, label):
        label = tokenizer.tokenize(label)
        label = label.to_tensor()
        if is_sequence_padding:
            label, mask = tftx.pad_model_inputs(label, max_seq_length=max_seq_length)



        return image, label

    return tokenize_label
