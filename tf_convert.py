import tensorflow as tf
import sys
sys.path.append('/home/max/Downloads/MTCNN/models/research')
from object_detection.utils import dataset_util
from PIL import Image
import glob
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_face_tf_example(encoded_face_iamge_data):
    """"creates a tf.Example proto from sample face image

    Args:
       encoded_faace_image_data: The jpg encoded data of the face image.

    Returns:
        example: The created tf.Example.
    """

    image_list = []
    for filename in glob.glob('/home/max/Downloads/train/lfw_5590'):
        im=Image.open(filename)
        image_list.append(im)

    
    data = pd.read_csv('/home/max/Downloads/train/trainImageList.txt', sep=" ", header=None)
    data.columns = ["filename", "xmins", "xmaxs", "ymins", "ymaxs", "x_left_eye", "y_left_eye", "x_right_eye", "y_right_eye", 
                    "x_nose", "y_nose", "x_left_mouth", "y_left_mouth", "x_right_mouth", "y_right_mouth"]



    image_format = b'jpg'
    height = 250
    width = 250
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    x_left_eye = []
    y_left_eye = []
    x_right_eye = []
    y_right_eye = []
    x_nose = []
    y_nose = []
    x_left_mouth = []
    y_left_mouth = []
    x_right_mouth = []
    y_right_mouth = []
    classes_text = [] 
    classes = []

    tf_example = tf.train.Example(features = tf.train.Features(feature={
        'image/height': dataset_util.int64_fature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename':dataset_util.int64_feature(filename),
        'image/source_id':dataset_util.bytes_feature(filename),
        'image/encoded':dataset_util.bytes_feature(encoded_face_iamge_data),
        'image/format':dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin':dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':dataset_util.float_list_feature(ymaxs),
        'image/object/landmark/x_left_eye': dataset_util.float_list_feature(x_left_eye),
        'image/object/landmark/y_left_eye': dataset_util.float_list_feature(y_left_eye),
        'image/object/landmark/x_right_eye': dataset_util.float_list_feature(x_right_eye),
        'image/object/landmark/y_right_eye': dataset_util.float_list_feature(y_right_eye),
        'image/object/landmark/x_nose': dataset_util.float_list_feature(x_nose),
        'image/object/landmark/y_nose': dataset_util.float_list_feature(y_nose),
        'image/object/landmark/x_left_mouth': dataset_util.float_list_feature(x_left_mouth),
        'image/object/landmark/y_left_mouth': dataset_util.float_list_feature(y_left_mouth),
        'image/object/landmark/x_right_mouth': dataset_util.float_list_feature(x_right_mouth),
        'image/object/landmark/y_right_mouth': dataset_util.float_list_feature(y_right_mouth),
        'image/object/class/text':dataset_util.float_list_feature(classes_text),
        'image/object/class/label':dataset_util.bytes_list_feature(classes),
    }))
    return tf_example

    def main(_):
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
        
        #TODO: write code to read the dataset to examples variable

        for example in examples:
            tf_example = create_face_tf_example(example)
            writer.write(tf_example.SerializeToString())
        
        writer.close()

    if __name__ = '__main__':
        tf.app.run()
