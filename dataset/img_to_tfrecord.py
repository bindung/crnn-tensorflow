import os, math, io, sys
sys.path.append("..")
import tensorflow as tf
from dataset.utils import _get_output_filename, int64_feature, bytes_feature
from PIL import Image
import class_util

format = 'jpeg'


def img_to_tfrecord(image_dir, gt_file, tf_filename):
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for l in open(gt_file):
            s = l.split(' ')
            imname = s[0]
            imname = imname[:-1]
            text = s[1].strip()
            text = text[1:-1]
            labels = [class_util.char_to_label(c) for c in text]

            image = Image.open(os.path.join(image_dir, imname))
            image = image.convert('RGB')
            image = image.resize((32, 100))
            b = io.BytesIO()
            image.save(b, format)
            example = tf.train.Example(features=tf.train.Features(feature={"labels": int64_feature(labels),
                                                                           'image/height': int64_feature(32),
                                                                           'image/width': int64_feature(100),
                                                                           "image/encoded": bytes_feature(b.getvalue()),
                                                                           'image/format': bytes_feature(format)}))
            tfrecord_writer.write(example.SerializeToString())
            print("OK ... {}, {}".format(imname, text))


image_dir = "../train_data"
gt_file = "../train_data/gt.txt"
output_file = "../train_data/train.tfrecord"
img_to_tfrecord(image_dir, gt_file, output_file)
