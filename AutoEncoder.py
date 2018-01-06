import tensorflow as tf
import numpy as np
import random

import enum

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
    read_threads = 1
    outputWidth = nWidth
    outputHeight = nHeight
    
    # loss function with L1 distance
    @staticmethod
    def L1(output, target):
        return tf.reduce_sum(tf.abs(target-output))
    
    # loss function with L2 distance
    @staticmethod
    def L2(output, target):
        return tf.reduce_sum(tf.square(target-output))
    
    def __init__(self, training_csv_file_name, **options):
        # options by argument
        self.batch_size = options.get('batch_size', 1)
        self.is_data_augmentation = options.get('is_data_augmentation', True)
        # Option to skip conecctions between corresponding layers of encoder and decoder as in U-net
        self.is_skip_connection = options.get('is_skip_connection', True)
        self.loss_function = options.get('loss_function', AutoEncoder.L1)
        
        isDebug = True
        if isDebug:
            print("batch_size : {0}".format(self.batch_size))
            print("data_augmentation : {0}".format(self.is_data_augmentation))
            print("skip_connection : {0}".format(self.is_skip_connection))
            print("loss_function : {0}".format(self.loss_function))
        
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()
            self.prepare_batch(training_csv_file_name)

    def prepare_model(self):
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, AutoEncoder.nPixels])
            x_image = tf.cast(tf.reshape(x, [self.batch_size, AutoEncoder.nWidth, AutoEncoder.nHeight, 1]), tf.float32)

            t = tf.placeholder(tf.float32, [None, AutoEncoder.nPixels])
            t_image = tf.reshape(t, [self.batch_size, AutoEncoder.nWidth, AutoEncoder.nHeight, 1])

            # keep probabilities for dropout layer
            keep_prob = tf.placeholder(tf.float32)
            keep_all = tf.constant(1.0, dtype=tf.float32)

            ## Data Augmentation
            if self.is_data_augmentation:
                x_tmp_array = []
                t_tmp_array = []
                for i in range(self.batch_size):
                    x_tmp = x_image[i, :, :, :]
                    t_tmp = t_image[i, :, :, :]

                    # flip each images left right and up down randomly
                    rint = random.randint(0, 2)
                    if rint%2 != 0:
                        x_tmp = tf.image.flip_left_right(x_tmp)
                        t_tmp = tf.image.flip_left_right(t_tmp)

                    rint = random.randint(0, 4)
                    # Some images has meaning in vertical direction,
                    # so images are flipped vertically in lower probability than horizontal flipping
                    if rint%4 == 0:
                        x_tmp = tf.image.flip_up_down(x_tmp)
                        t_tmp = tf.image.flip_up_down(t_tmp)

                    rint = random.randint(0, 4)
                    # Some images has meaning in vertical direction,
                    # so images are transposed in lower probability than horizontal flipping
                    if rint%4 == 0:
                        x_tmp = tf.image.transpose_image(x_tmp)
                        t_tmp = tf.image.transpose_image(t_tmp)

                    x_tmp_array.append(tf.expand_dims(x_tmp, 0))
                    t_tmp_array.append(tf.expand_dims(t_tmp, 0))

                x_image = tf.concat(x_tmp_array, axis=0)
                t_image = tf.concat(t_tmp_array, axis=0)
            
            self.x_image = x_image

        # Encoding function for AutoEncoder
        def encode(batch_input, out_channels, stride, filter_size):
            with tf.variable_scope("encode"):
                in_channels = batch_input.get_shape()[3]

                filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
                # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
                #     => [batch, out_height, out_width, out_channels]
                
                # padding if needed with the values of filter_size and stride to fit output size to out_channels
                pad_size = int(filter_size - stride)
                if pad_size > 0:
                    pad0 = pad_size//2
                    pad1 = pad_size//2 + pad_size%2
                    batch_input = tf.pad(batch_input, [[0, 0], [pad0, pad1], [pad0, pad1], [0, 0]], mode="CONSTANT")

                conved = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="VALID")
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
        def decode(batch_input, out_channels, filter_size):
            with tf.variable_scope("decode"):
                batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
                filter = tf.get_variable("filter", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
                # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
                #     => [batch, out_height, out_width, out_channels]
                conved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
                return conved

        # List to contain each layer of AutoEncoder
        layers = []

        # 3 if used for color image
        num_channels = 1
        out_channels_base = 32
        encode_output_channels = [
            (out_channels_base    , 16), # encoder_1: [batch_size, 512, 512, num_channels] => [batch_size, 256, 256, out_channels_base * 2]
            (out_channels_base * 2, 8),  # encoder_2: [batch_size, 256, 256, out_channels_base] => [batch_size, 128, 128, out_channels_base * 2]
            (out_channels_base * 4, 4),  # encoder_3: [batch_size, 128, 128, out_channels_base] => [batch_size, 64, 64, out_channels_base * 2]
            (out_channels_base * 8, 4),  # encoder_4: [batch_size, 64, 64, out_channels_base * 2] => [batch_size, 32, 32, out_channels_base * 4]
            (out_channels_base * 8, 4),  # encoder_5: [batch_size, 32, 32, out_channels_base * 4] => [batch_size, 16, 16, out_channels_base * 8]
            (out_channels_base * 8, 4),  # encoder_6: [batch_size, 16, 16, out_channels_base * 8] => [batch_size, 8, 8, out_channels_base * 8]
            (out_channels_base * 8, 4),  # encoder_7: [batch_size, 8, 8, out_channels_base * 8] => [batch_size, 4, 4, out_channels_base * 8]
            (out_channels_base * 8, 4),  # encoder_8: [batch_size, 4, 4, out_channels_base * 8] => [batch_size, 2, 2, out_channels_base * 8]
            #(out_channels_base * 8, 2)  # encoder_9: [batch_size, 2, 2, out_channels_base * 8] => [batch_size, 1, 1, out_channels_base * 8]
        ]

        for encoder_index, (out_channels, filter_size) in enumerate(encode_output_channels):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                if encoder_index == 0:
                    output = encode(x_image, out_channels, 2, filter_size)
                else:
                    rectified = lrelu(layers[-1], 0.2)
                    # [batch_size, height, width, in_channels] => [batch_size, height/2, width/2, out_channels]
                    encoded = encode(rectified, out_channels, 2, filter_size)
                    output = batchnorm(encoded)
                layers.append(output)
                print(output)

        decode_output_channels = [
            #(out_channels_base * 8, keep_prob, 2), # decoder_9: [batch_size, 1, 1, out_channels_base * 8] => [batch_size, 2, 2, out_channels_base * 8 * 2]
            (out_channels_base * 8, keep_prob, 4),  # decoder_8: [batch_size, 2, 2, out_channels_base * 8 * 2] => [batch_size, 4, 4, out_channels_base * 8 * 2]
            (out_channels_base * 8, keep_prob, 4),  # decoder_7: [batch_size, 4, 4, out_channels_base * 8 * 2] => [batch_size, 8, 8, out_channels_base * 8 * 2]
            (out_channels_base * 8, keep_all, 4),  # decoder_6: [batch_size, 8, 8, out_channels_base * 8 * 2] => [batch_size, 16, 16, out_channels_base * 8 * 2]
            (out_channels_base * 8, keep_all, 4),  # decoder_5: [batch_size, 16, 16, out_channels_base * 8 * 2] => [batch_size, 32, 32, out_channels_base * 4 * 2]
            (out_channels_base * 4, keep_all, 4),  # decoder_4: [batch_size, 32, 32, out_channels_base * 4 * 2] => [batch_size, 64, 64, out_channels_base * 2 * 2]
            (out_channels_base * 2, keep_all, 4),  # decoder_3: [batch_size, 64, 64, out_channels_base * 4 * 2] => [batch_size, 128, 128, out_channels_base * 2 * 2]
            (out_channels_base    , keep_all, 8),  # decoder_2: [batch_size, 128, 128, out_channels_base * 2 * 2] => [batch_size, 256, 256, out_channels_base * 2]
            (num_channels         , keep_all, 16), # decoder_1: [batch, 256, 256, out_channels_base * 2] => [batch, 512, 512, num_channels]
        ]
        
        num_encoder_layers = len(layers)
        for decoder_index, (out_channels, dropout_keep_prob, filter_size) in enumerate(decode_output_channels):
            skip_layer = num_encoder_layers - decoder_index - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_index == 0:
                    # Even if "skip connection", first layer in decode layers is connected only from encode layer
                    input = layers[-1]
                else:
                    if self.is_skip_connection == True:
                        # Concat output from encoder layer to keep detailed information
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    else:
                        input = layers[-1]

                rectified = tf.nn.relu(input)
                
                # [batch_size, height, width, in_channels] => [batch_size, height*2, width*2, out_channels]
                output = decode(rectified, out_channels, filter_size)

                if decoder_index != num_encoder_layers-1:
                    output = batchnorm(output)
                #else:
                    # final decoder
                    #output = tf.tanh(output)

                # dropout layer
                output = tf.cond(dropout_keep_prob < 1.0, lambda: tf.nn.dropout(output, keep_prob=dropout_keep_prob), lambda: output)

                layers.append(output)
                print(output)


        output = layers[-1]

        with tf.name_scope("Optimizer"):
            ## Apply loss function (difference between training data and predicted data), and learning algorithm.
            t_compare = t_image
            loss = self.loss_function(output, t_compare)
            train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

        tf.summary.scalar("loss", loss)
        #tf.histogram_summary("Convolution_1:biases", b_conv1)
        
        self.x = x
        self.t = t
        self.keep_prob = keep_prob
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

    def prepare_batch(self, training_csv_file_name):
        image_batch, depth_batch = self.input_pipeline([training_csv_file_name], self.batch_size, AutoEncoder.read_threads)
        
        self.image_batch = image_batch
        self.depth_batch = depth_batch
