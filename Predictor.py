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

ae = AutoEncoder("predictList.csv", is_data_augmentation=False)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=ae.sess)

with ae.sess as sess:
    ae.saver.restore(sess, "./saved_session/s-36000")
    
    try:
        while not coord.should_stop():
            image_in_csv, depth_data = ae.sess.run([ae.image_batch, ae.depth_batch])
            image_in_csv = image_in_csv.reshape((ae.batch_size, AutoEncoder.nPixels))

            out, x_input = ae.sess.run([ae.output, ae.x_image], feed_dict={ae.x:image_in_csv, ae.keep_prob:1.0})

            draw_image([x_input, out],
                       ["Input Image", "Output Image"],
                       ae.batch_size, ae.outputWidth, ae.outputHeight)
            coord.request_stop()

    except tf.errors.OutOfRangeError:
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

ae.sess.close()

print("Predictor End")
