import os
import scipy.misc
import numpy as np
from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
import tensorflow as tf

# 定义各种训练所需要的配置信息，参数说明都写有了

# 这里的batch_size有个bug，据说设置为 64 或者 16 才能在训练的时候正常生成Sampls图片,报错原因和数量有关，没认真研究
# 最终导致错误的位置在utils.py中倒数第二句:assert manifold_h * manifold_w == num_images

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 100, "The size of image to use (will be center cropped). [100]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 100, "The size of the output images to produce [100]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "image", "The name of dataset [image2, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):

    # 打印上面的配置参数
    pp.pprint(flags.FLAGS.__flags)
    # 定义输入和输出的宽度
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height
    # 创建检查点和存放输出样本的文件夹，没有就创建
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # 目测是使配置参数生效

    run_config = tf.ConfigProto()
    # 使用gpu
    run_config.gpu_options.allow_growth=True
    # 在会话中训练
    with tf.Session(config=run_config) as sess:
    #  如果数据集是mnist的操作
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                z_dim=FLAGS.generate_test_images,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir)
        else:

            dcgan = DCGAN(
                sess,
                # 获取输入宽度，高
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                # 获取输出宽度，高
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                # batch_size 和 样本数量的值
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                # 噪声的维度
                z_dim=FLAGS.generate_test_images,
                # 获取数据集名字
                dataset_name=FLAGS.dataset,
                # 输入的文件格式(.jpg)
                input_fname_pattern=FLAGS.input_fname_pattern,
                # 是否使用中心裁剪
                crop=FLAGS.crop,
                # 检查点,样本储存,数据集的文件夹路径
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir)

            # 输出所有参数??
            show_all_variables()
            # 训练模式为True
        if FLAGS.train:
            # 调用函数执行训练
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")


        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        OPTION = 1
        # 可视化操作
        visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
