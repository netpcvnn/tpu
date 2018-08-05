# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import multiprocessing
import os
from absl import flags
import numpy as np
# import PIL.Image
import PIL.Image
from PIL import Image


import dataset_util
# from pycocotools import mask
# from research.object_detection.utils import dataset_util
# from research.object_detection.utils import label_map_util

import tensorflow as tf

# flags.DEFINE_boolean('include_masks', False,
#                      'Whether to include instance segmentations masks '
#                      '(PNG encoded) in the result. default: False.')
# flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
# flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
# flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
# flags.DEFINE_string('train_annotations_file', '',
#                     'Training annotations JSON file.')
# flags.DEFINE_string('val_annotations_file', '',
#                     'Validation annotations JSON file.')
# flags.DEFINE_string('testdev_annotations_file', '',
#                     'Test-dev annotations JSON file.')
# flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

# FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(
    image,
    annotations_list,
    image_dir,
    category_index,
    include_masks=False):
    
    # image_height = image[2]
    # image_width = image[1]
    # filename = image[0]# TODO(user): Populate the following variables from your example.
    # print(image)
    height = image['height'] # Image height
    width = image['width'] # Image width
    filename = image['filename'] # Filename of the image. Empty if image is not from file

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_image_io = io.BytesIO(encoded_jpg) # Encoded image bytes
    image = PIL.Image.open(encoded_image_io)
    only_file_name, image_format = os.path.splitext(filename)

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per bo)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    # print(len(annotations_list))
    for annotation in annotations_list:
        # print(annotation)
        xmins.append(annotation['xmin']/width)
        xmaxs.append(annotation['xmax']/width)
        ymins.append(annotation['ymin']/height)
        ymaxs.append(annotation['ymax']/height)
        classes_text.append(annotation['label_text'].encode('utf8'))
        classes.append(annotation['label'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
def _pool_create_tf_example(args):
  return create_tf_example(*args)
def _create_tf_record_from_retina_pytorch_annotation(
    annotations_file, image_dir,
    output_path, num_shards):

    # self.root = root
    # self.input_size = input_size

    images = []
    boxes = []
    labels = []
    annotations=[]

    # self.encoder = DataEncoder()

    with tf.gfile.GFile(annotations_file) as f:
        lines = f.readlines()
        num_samples = len(lines)

    for line in lines:
        splited = line.strip().split()
        # images.append(splited[0])
        full_path = os.path.join(image_dir, splited[0])
        im=Image.open(full_path)
        num_boxes = (len(splited) - 1) // 5
        box = []
        label = []
        name={}
        name['filename']= splited[0]
        name['width']= int(im.width)
        name['height']=int(im.height)
        annotations=[]
        # name.append([splited[0], float(im.width), float(im.height)])
        for i in range(num_boxes):
            xmin = splited[1+5*i]
            ymin = splited[2+5*i]
            xmax = splited[3+5*i]
            ymax = splited[4+5*i]
            c = splited[5+5*i]
            # box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
            # label.append(int(c))

            annotation={}
            annotation['xmin']= float(xmin)
            annotation['xmax']= float(xmax)
            annotation['ymin']= float(ymin)
            annotation['ymax']= float(ymax)
            annotation['label']= int(c)
            annotation['label_text']= str(c)

            # annotation.append([float(xmin),float(ymin),float(xmax),float(ymax), int(c)])
            # print(annotation)
            # print(num_boxes)
            # print(annotation)
            annotations.append(annotation)
        name['annotation']=annotations

        images.append(name)
        
        # self.boxes.append(torch.Tensor(box))
        # self.labels.append(torch.LongTensor(label))


    include_masks= False
    category_index = 1
    tf.logging.info('writing to output path: %s', output_path)
    writers = [ tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards)) for i in range(num_shards)
    ]
    writer = tf.python_io.TFRecordWriter(output_path+ 'test.tfrecord')

    for i in range(len(images)):
        tf_example = create_tf_example(images[i], images[i]['annotation'], image_dir, category_index,include_masks)
        writer.write(tf_example.SerializeToString())
    writer.close()

    # pool = multiprocessing.Pool()
    # total_num_annotations_skipped = 0
    # for idx, (_, tf_example, num_annotations_skipped) in enumerate(
    #     pool.imap(_pool_create_tf_example,
    #                 [(images[i], annotations[i], image_dir,
    #                 category_index, include_masks)
    #                 for i in range(len(images))])):
    #     if idx % 100 == 0:
    #         tf.logging.info('On image %d of %d', idx, len(images))

    #     # total_num_annotations_skipped += num_annotations_skipped
    #     # writers[idx % num_shards].write(tf_example.SerializeToString())

    # pool.close()
    # pool.join()
def main():

    print("Start functions")

    # assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    # assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    # assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    # assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'
    output_dir='/home/liem/Downloads'
    train_annotations_file='/home/liem/Downloads/test_data.txt'
    train_img_dir='/home/liem/Downloads/test'

    # if not tf.gfile.IsDirectory(output_dir):
    #     tf.gfile.MakeDirs(output_dir)
    train_output_path = os.path.join(output_dir, 'train')
    # val_output_path = os.path.join(FLAGS.output_dir, 'val')
    # testdev_output_path = os.path.join(FLAGS.output_dir, 'test-dev')


    _create_tf_record_from_retina_pytorch_annotation(
        train_annotations_file,
        train_img_dir,
        train_output_path,
        num_shards=1)

if __name__ == '__main__':
    main()
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()




