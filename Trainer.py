## Trainer for AutoEncoder learning

import tensorflow as tf
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

print("start")

# function to draw image
def draw_image(image_list, caption_list, batch_size, width, height):
    num_image = len(image_list)
    for i in range(num_image):
        image_list[i] = image_list[i].reshape((batch_size, width, height))

    for i in range(batch_size):
        fig = plt.figure(figsize=(15,15))
        for j in range(num_image):
            subplot = fig.add_subplot(1,num_image,j+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(caption_list[j]+str(i))
            # image_list[j] is each images with 4-D(first dimension is batch_size)
            subplot.imshow(image_list[j][i], vmin=0, vmax=255, cmap=plt.cm.gray, interpolation="nearest")

ae = AutoEncoder('dataList.csv', 8, True)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=ae.sess)
scaler = MinMaxScaler(feature_range=(0,1))

# If you want to resume learning, please set start step number larger than 0
start = 0
# loop counter
i = start

# number of loops to report loss value while learning
n_report_loss_loop = 2

# number of loops to report predicted image while learning
# this must be multiple number of n_report_loss_loop
n_report_image_loop = 10

# number of all loops for learning
n_all_loop = 2000000


print("Start Learning loop")
with ae.sess as sess:
    
    if start > 0:
        print("Resume from session files")
        ae.saver.restore(sess, "./saved_session/s-" +str(start))
    
    try:
        while not coord.should_stop():
            i += 1
            # Run training steps or whatever
            image_data, depth_data = ae.sess.run([ae.image_batch, ae.depth_batch])
            image_data = image_data.reshape((ae.batch_size, AutoEncoder.nPixels))
            #image_data = scaler.fit_transform(image_data)
            depth_data = depth_data.reshape((ae.batch_size, AutoEncoder.nPixels))
            
            ae.sess.run([ae.train_step], feed_dict={ae.x:image_data, ae.t:depth_data, ae.keep_prob:0.5})
            if i == n_all_loop:
                coord.request_stop()

            #TODO:Split data into groups for cross-validation
            if i==start+1 or i % n_report_loss_loop == 0:
                loss_vals = []
                loss_val, t_cmp, out, summary, x_input = ae.sess.run([ae.loss, ae.t_compare, ae.output, ae.summary, ae.x_image],
                                                            feed_dict={ae.x:image_data, ae.t:depth_data, ae.keep_prob:1.0})
                loss_vals.append(loss_val)
                loss_val = np.sum(loss_vals)
                ae.saver.save(ae.sess, './saved_session/s', global_step=i)
                print ('Step: %d, Loss: %f @ %s' % (i, loss_val, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                if i==start+1 or i % n_report_image_loop == 0:
                    x_input = tf.reshape(x_input, [ae.batch_size, ae.outputWidth, ae.outputHeight])
                    t_cmp = tf.reshape(t_cmp, [ae.batch_size, ae.outputWidth, ae.outputHeight])
                    out = tf.reshape(out, [ae.batch_size, ae.outputWidth, ae.outputHeight])
                    draw_image([x_input.eval(session=ae.sess), out.eval(session=ae.sess), t_cmp.eval(session=ae.sess)],
                               ["Input Image", "Predicted Result", "Ground Truth"],
                               ae.batch_size, ae.outputWidth, ae.outputHeight)
                    ae.writer.add_summary(summary, i)

    except tf.errors.OutOfRangeError:
        print('Done training')
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

ae.sess.close()

print("end")
