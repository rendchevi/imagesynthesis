import numpy as np
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf

def batch(data, batch_size):
    for ndx in range(0, len(data), batch_size):
        yield data[ndx:min(ndx + batch_size, len(data))]

def process_images(image_files, target_shape):
    # Read the images
    images = np.array([resize((imread(image_file) - 127.5) / 127.5, target_shape) for image_file in image_files])
    # Expand the dimensions
    if len(images.shape) != 4:
        images = np.expand_dims(images, axis = -1)
    # Convert to Tensor
    images = tf.convert_to_tensor(images, dtype = tf.float32)
    # Convert to Variable
    images = tf.Variable(images, trainable = True)

    return images

def deprocess_images(images):
    # Convert to numpy array
    images = images.numpy()
    # Squeeze the dimensions
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis = -1)
    # De-normalize image
    images = np.array([(127.5 * image) + 127.5 for image in images]).astype(np.uint8)

    return images

def fetch_dataset(data_path, batch_size, image_size, color_mode):
    # Generate train dataset
    train_datagen = tf.keras.preprocessing.image_dataset_from_directory(data_path,
                                                                        image_size = image_size,
                                                                        batch_size = batch_size,
                                                                        color_mode = color_mode)

    # Normalize training dataset
    norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
    train_dataset = train_datagen.map(lambda x, y: (norm_layer(x), y))

    return train_dataset
