## Executor for AutoEncoder learning

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

nn = AutoEncoder('dataList.csv', 4)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=nn.sess)
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
with nn.sess as sess:
    
    if start > 0:
        print("Resume from session files")
        nn.saver.restore(sess, "./saved_session/s-" +str(start))
    
    try:
        while not coord.should_stop():
            i += 1
            # Run training steps or whatever
            image_data, depth_data = nn.sess.run([nn.image_batch, nn.depth_batch])
            image_data = image_data.reshape((nn.batch_size, AutoEncoder.nPixels))
            #image_data = scaler.fit_transform(image_data)
            depth_data = depth_data.reshape((nn.batch_size, AutoEncoder.nPixels))
            
            nn.sess.run([nn.train_step], feed_dict={nn.x:image_data, nn.t:depth_data, nn.keep_prob:0.5})
            if i == n_all_loop:
                coord.request_stop()

            #TODO:Split data into groups for cross-validation
            if i==start+1 or i % n_report_loss_loop == 0:
                loss_vals = []
                loss_val, t_cmp, out, summary, x_input = nn.sess.run([nn.loss, nn.t_compare, nn.output, nn.summary, nn.x_image],
                                                            feed_dict={nn.x:image_data, nn.t:depth_data, nn.keep_prob:1.0})
                loss_vals.append(loss_val)
                loss_val = np.sum(loss_vals)
                nn.saver.save(nn.sess, './saved_session/s', global_step=i)
                print ('Step: %d, Loss: %f @ %s' % (i, loss_val, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                if i==start+1 or i % n_report_image_loop == 0:
                    x_input = tf.reshape(x_input, [nn.batch_size, nn.outputWidth, nn.outputHeight])
                    t_cmp = tf.reshape(t_cmp, [nn.batch_size, nn.outputWidth, nn.outputHeight])
                    out = tf.reshape(out, [nn.batch_size, nn.outputWidth, nn.outputHeight])
                    draw_image([x_input.eval(session=nn.sess), out.eval(session=nn.sess), t_cmp.eval(session=nn.sess)],
                               ["Input Image", "Predicted Result", "Ground Truth"],
                               nn.batch_size, nn.outputWidth, nn.outputHeight)
                    nn.writer.add_summary(summary, i)

    except tf.errors.OutOfRangeError:
        print('Done training')
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

nn.sess.close()

print("end")
