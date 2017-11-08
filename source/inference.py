import tensorflow as tf
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np



num_images_in_validation_set = 60

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

model = load_graph('./export/constant_graph_weights.pb')


# for n in model.as_graph_def().node:
#     print n.name


# img = plt.imread('./dataset/val/WW/139.jpg')
# img = np.reshape(img,[1,256,256,3])
# img = img.astype(float)
# img = img/255.0
# input_node = model.get_tensor_by_name("prefix/conv2d_7_input_2:0")
# output_node = model.get_tensor_by_name("prefix/output_node0:0")

# with tf.Session(graph=model) as sess:
#         # Note: we didn't initialize/restore anything, everything is stored in the graph_def
#         y_out = sess.run(output_node, feed_dict={
#             input_node: img
#         })
#         print(y_out)


test_data_dir = 'dataset/val'

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='binary',
    shuffle=False)


preds = new_model.predict_generator(generator=test_generator, steps=num_images_in_validation_set)

pred_cl = preds > 0.5
pred_cl = pred_cl.astype(np.int)
pred_for_file = zip(test_generator.filenames, pred_cl, preds)

for i in pred_for_file:
    print(i)







import cv2
import os
import numpy as np
import csv


images = 'dataset/train/regular'

def convert_image_to_vector(csv_filename, img):
    histr_blue = cv2.calcHist([img],[0],None,[256],[0,256])
    histr_green = cv2.calcHist([img],[1],None,[256],[0,256])
    histr_red = cv2.calcHist([img],[2],None,[256],[0,256])

    histr_blue = np.reshape(histr_blue, (256, ))
    histr_green = np.reshape(histr_green, (256, ))
    histr_red = np.reshape(histr_red, (256, ))

    stack = np.concatenate((histr_blue, histr_green), axis=0)
    stack = np.concatenate((stack, histr_red), axis=0)

    with open(os.path.join(images, csv_filename), "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(stack)


def main():
    for file in os.listdir(images):
        if file.startswith('.'):
            continue
        img = cv2.imread(os.path.join(images, file))
        
        convert_image_to_vector("regular.csv", img)


main()




