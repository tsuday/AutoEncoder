## Class definition for AutoEncoder

import tensorflow as tf
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

random.seed(a=123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

## Learning Model
class AutoEncoder:
    # input image size
    nWidth = 512
    nHeight = 512
    nPixels = nWidth * nHeight
    batch_size = 8
    read_threads = 1
    outputWidth = nWidth
    outputHeight = nHeight

    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()
            self.prepare_batch()

    def prepare_model(self):
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, AutoEncoder.nPixels])
            x_image = tf.cast(tf.reshape(x, [AutoEncoder.batch_size, AutoEncoder.nWidth, AutoEncoder.nHeight, 1]), tf.float32)

            t = tf.placeholder(tf.float32, [None, AutoEncoder.nPixels])
            t_image = tf.reshape(t, [AutoEncoder.batch_size, AutoEncoder.nWidth, AutoEncoder.nHeight, 1])
            
            ## Data Augmentation
            # flip each images left right and up down randomly
            x_tmp_array = []
            t_tmp_array = []
            for i in range(AutoEncoder.batch_size):
                x_tmp = x_image[i, :, :, :]
                t_tmp = t_image[i, :, :, :]

                rint = random.randint(0, 2)
                if rint%2 != 0:
                    x_tmp = tf.image.flip_left_right(x_tmp)
                    t_tmp = tf.image.flip_left_right(t_tmp)

                rint = random.randint(0, 2)
                if rint%2 != 0:
                    x_tmp = tf.image.flip_up_down(x_tmp)
                    t_tmp = tf.image.flip_up_down(t_tmp)

                x_tmp_array.append(tf.expand_dims(x_tmp, 0))
                t_tmp_array.append(tf.expand_dims(t_tmp, 0))

            x_image = tf.concat(x_tmp_array, axis=0)
            t_image = tf.concat(t_tmp_array, axis=0)
            
            self.x_image = x_image

        # Encoding function for AutoEncoder
        def encode(batch_input, out_channels, stride):
            with tf.variable_scope("encode"):
                in_channels = batch_input.get_shape()[3]
                filter_size = 4

                filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
                # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
                #     => [batch, out_height, out_width, out_channels]
                padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
                return conved

        # Leaky ReLU
        def lrelu(x, a):
            with tf.name_scope("LeakyReLU"):
                x = tf.identity(x)
                # leak[a*x/2 - a*abs(x)/2] + linear[x/2 + abs(x)/2]
                return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

        # Batch Normalization
        def batchnorm(input):
            with tf.variable_scope("BatchNormalization"):
                input = tf.identity(input)

                channels = input.get_shape()[3]
                offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer)
                scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
                mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
                variance_epsilon = 1e-5
                normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
                return normalized

        # Decoding function for AutoEncoder
        def decode(batch_input, out_channels):
            with tf.variable_scope("decode"):
                filter_size = 4
                batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
                filter = tf.get_variable("filter", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
                # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
                #     => [batch, out_height, out_width, out_channels]
                conved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
                return conved

        # List to contain each layer of AutoEncoder
        layers = []

        out_channels_base = 32
        with tf.name_scope("encoder_1"):
            # encoder_1: [batch_size, 512, 512, out_channels_base] => [batch_size, 256, 256, out_channels_base * 2]
            otpt = encode(x_image, out_channels_base, stride=2)
            layers.append(otpt)

        encode_output_channels = [
            out_channels_base * 2,  # encoder_2: [batch_size, 256, 256, out_channels_base] => [batch_size, 128, 128, out_channels_base * 2]
            out_channels_base * 4,  # encoder_3: [batch_size, 128, 128, out_channels_base] => [batch_size, 64, 64, out_channels_base * 2]
            out_channels_base * 8,  # encoder_4: [batch_size, 64, 64, out_channels_base * 2] => [batch_size, 32, 32, out_channels_base * 4]
            out_channels_base * 8,  # encoder_5: [batch_size, 32, 32, out_channels_base * 4] => [batch_size, 16, 16, out_channels_base * 8]
            out_channels_base * 8,  # encoder_6: [batch_size, 16, 16, out_channels_base * 8] => [batch_size, 8, 8, out_channels_base * 8]
            out_channels_base * 8,  # encoder_7: [batch_size, 8, 8, out_channels_base * 8] => [batch_size, 4, 4, out_channels_base * 8]
            out_channels_base * 8,  # encoder_8: [batch_size, 4, 4, out_channels_base * 8] => [batch_size, 2, 2, out_channels_base * 8]
            #out_channels_base * 8, # encoder_9: [batch_size, 2, 2, out_channels_base * 8] => [batch_size, 1, 1, out_channels_base * 8]
        ]

        for out_channels in encode_output_channels:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch_size, height, width, in_channels] => [batch_size, height/2, width/2, out_channels]
                encoded = encode(rectified, out_channels, stride=2)
                output = batchnorm(encoded)
                layers.append(output)

        decode_output_channels = [
            #(out_channels_base * 8, 0.5), # decoder_9: [batch_size, 1, 1, out_channels_base * 8] => [batch_size, 2, 2, out_channels_base * 8 * 2]
            (out_channels_base * 8, 0.5),  # decoder_8: [batch_size, 2, 2, out_channels_base * 8 * 2] => [batch_size, 4, 4, out_channels_base * 8 * 2]
            (out_channels_base * 8, 0.5),  # decoder_7: [batch_size, 4, 4, out_channels_base * 8 * 2] => [batch_size, 8, 8, out_channels_base * 8 * 2]
            (out_channels_base * 8, 0.0),  # decoder_6: [batch_size, 8, 8, out_channels_base * 8 * 2] => [batch_size, 16, 16, out_channels_base * 8 * 2]
            (out_channels_base * 8, 0.0),  # decoder_5: [batch_size, 16, 16, out_channels_base * 8 * 2] => [batch_size, 32, 32, out_channels_base * 4 * 2]
            (out_channels_base * 4, 0.0),  # decoder_4: [batch_size, 32, 32, out_channels_base * 4 * 2] => [batch_size, 64, 64, out_channels_base * 2 * 2]
            (out_channels_base * 2, 0.0),  # decoder_3: [batch_size, 64, 64, out_channels_base * 4 * 2] => [batch_size, 128, 128, out_channels_base * 2 * 2]
            (out_channels_base, 0.0),      # decoder_2: [batch_size, 128, 128, out_channels_base * 2 * 2] => [batch_size, 256, 256, out_channels_base * 2]
        ]
        
        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(decode_output_channels):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # Connected only from encode layer
                    input = layers[-1]
                else:
                    # Also has skip connection from encoder layer
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                
                # [batch_size, height, width, in_channels] => [batch_size, height*2, width*2, out_channels]
                output = decode(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        final_output_channels = 1 # 3 if used for color image
        # decoder_1: [batch, 256, 256, out_channels_base * 2] => [batch, 512, 512, final_output_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = decode(rectified, final_output_channels)
            layers.append(output)

        output = layers[-1]

        for l in layers:
            print(l)

        with tf.name_scope("Optimizer"):
            ## Define loss(difference between training data and predicted data), and criteria for learning algorithm.

            t_compare = t_image
            loss = tf.reduce_sum(tf.square(t_compare-output))
            train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

        tf.summary.scalar("loss", loss)
        #tf.histogram_summary("Convolution_1:biases", b_conv1)
        
        self.x = x
        self.t = t
        self.train_step = train_step
        self.output = output
        self.t_compare = t_compare
        self.loss = loss
        
    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("board/learn_logs", sess.graph)
        
        self.sess = sess
        self.saver = saver
        self.summary = summary
        self.writer = writer

    # https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#batching
    def read_my_file_format(self, filename_queue):
        reader = tf.TextLineReader()
        key, record_string = reader.read(filename_queue)
        # "a" means representative value to indicate type for csv cell value.
        image_file_name, depth_file_name = tf.decode_csv(record_string, [["a"], ["a"]])

        image_png_data = tf.read_file(image_file_name)
        depth_png_data = tf.read_file(depth_file_name)
        # channels=1 means image is read as gray-scale
        image_decoded = tf.image.decode_png(image_png_data, channels=1)
        image_decoded.set_shape([512, 512, 1])
        depth_decoded = tf.image.decode_png(depth_png_data, channels=1)
        depth_decoded.set_shape([512, 512, 1])
        return image_decoded, depth_decoded

    def input_pipeline(self, filenames, batch_size, read_threads, num_epochs=None):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        min_after_dequeue = 2
        capacity = min_after_dequeue + 3 * batch_size

        example_list = [self.read_my_file_format(filename_queue) for _ in range(read_threads)]
        example_batch, label_batch = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch

    def prepare_batch(self):
        image_batch, depth_batch = self.input_pipeline(['dataList.csv'], AutoEncoder.batch_size, AutoEncoder.read_threads)
        
        self.image_batch = image_batch
        self.depth_batch = depth_batch
