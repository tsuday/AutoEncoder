### Predictor 
%matplotlib inline
import matplotlib.pyplot as plt

print("Predictor Start")

# function to draw image
def draw_image(image_list, caption_list, batch_size, width, height):
    num_image = len(image_list)
    for i in range(num_image):
        image_list[i] = image_list[i].reshape((batch_size, width, height))

    for i in range(batch_size):
        fig = plt.figure(figsize=(20,20))
        for j in range(num_image):
            subplot = fig.add_subplot(1,num_image,j+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(caption_list[j]+str(i))
            # image_list[j] is each images with 4-D(first dimension is batch_size)
            subplot.imshow(image_list[j][i], vmin=0, vmax=255, cmap=plt.cm.gray, interpolation="nearest")

nn = AutoEncoder("predictList.csv", 1, False)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=nn.sess)

with nn.sess as sess:
    nn.saver.restore(sess, "./saved_session/s-36000")
    
    try:
        while not coord.should_stop():
            image_in_csv, depth_data = nn.sess.run([nn.image_batch, nn.depth_batch])
            image_in_csv = image_in_csv.reshape((nn.batch_size, AutoEncoder.nPixels))

            out, x_input = nn.sess.run([nn.output, nn.x_image], feed_dict={nn.x:image_in_csv, nn.keep_prob:1.0})

            draw_image([x_input, out],
                       ["Input Image", "Output Image"],
                       nn.batch_size, nn.outputWidth, nn.outputHeight)
            coord.request_stop()

    except tf.errors.OutOfRangeError:
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

nn.sess.close()

print("Predictor End")
