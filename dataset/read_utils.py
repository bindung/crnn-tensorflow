import tensorflow as tf


def read_and_decode(filename, num_epochs):  # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature((), tf.string),
                                           'labels': tf.VarLenFeature(tf.int64)
                                       })  # return image and label

    img = tf.image.decode_png(features['image/encoded'])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # throw img tensor
    labels = tf.cast(features['labels'], tf.int32)  # throw label tensor
    return img, img.shape[1], labels


def inputs(batch_size, num_epochs, filename):
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        # Even when reading in multiple threads, share the filename
        # queue.
        img, width, label = read_and_decode(filename, num_epochs)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        sh_images, sh_width, sh_labels = tf.train.shuffle_batch(
            [img, width, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=100)

        return sh_images, sh_width, sh_labels


def preprocess_for_train(image, label, scope='crnn_preprocessing_train'):
    """Preprocesses the given image for training.
    """
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert image as a tf.float32 tensor
            image_s = tf.expand_dims(image, 0)
            tf.summary.image("image", image_s)

        image = tf.image.rgb_to_grayscale(image)
        tf.summary.image("gray", image)
        return image, label
