import numpy as np
import cv2
import time
import tensorflow as tf
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

# my laptop camera frame_rate = 12-13 fps

frame_eval_interval = 6

# dimensions of our images.
img_width, img_height = 384, 286

labels = ['go_down', 'go_left', 'go_right', 'go_up', 'perfect']

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph


def main():
    # load the model here
    model = load_graph('./export_2_classification/constant_graph_weights_new_dataset.pb')

    # for n in model.as_graph_def().node:
    #     print n.name

    input_node = model.get_tensor_by_name("prefix/conv2d_1_input_2:0")
    output_node = model.get_tensor_by_name("prefix/output_node0:0")



    cap = cv2.VideoCapture(0)

    frame_count = 0
    t0 = time.time()

    # live video capture starts here
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_count += 1

        if frame_count%frame_eval_interval == 0:
            # do the inference on this frame
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # resize the image to required dimensions
            gray = cv2.resize(gray, (img_width, img_height))
            gray = np.stack((gray,)*3)
            # reshape the image to required tensor dimension
            gray = np.reshape(gray, [1, img_width, img_height, 3])

            # then call inference on the gray frame            
            with tf.Session(graph=model) as sess:
                y_out = sess.run(output_node, feed_dict={
                    input_node: gray
                })
                # print(y_out)
                pred_class = np.argmax(y_out)
                pred_score = np.max(y_out)
                print(labels[pred_class], "score - {}".format(pred_score))
                

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    t1 = time.time()
    # live video capture ends here

    print('frame_count - ', frame_count)
    total = t1-t0
    print('time elapsed - ', total)
    frame_rate = frame_count/total
    print('frame rate - ', frame_rate)


main()