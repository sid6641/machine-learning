from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 384, 286

# change the directories accrding to requirements
train_data_dir = '../experiment/train'
validation_data_dir = '../experiment/val'

# train_data_dir = '../dataset/train'
# validation_data_dir = '../dataset/val'

num_classes = 5
# number of training examples (including all the classes)
nb_train_samples = 3042*5
nb_validation_samples = 15*5
epochs = 1
batch_size = 10


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save_weights('weight.h5')


# export mode starts here


# K.set_learning_phase(0)
# config = model.get_config()
# weights = model.get_weights()

# new_model = Sequential.from_config(config)
# new_model.set_weights(weights)

# new_model.save('2_classification_test_model.h5')


# weight_file_path = './2_classification_test_model.h5'
# num_output = 1
# prefix_output_node_names_of_final_network = 'output_node'
# export_path =  './export' # where to save the exported graph
# output_graph_name = 'constant_graph_weights.pb'

# K.set_learning_phase(0)
# pred = [None]*num_output
# pred_node_names = [None]*num_output
# net_model = load_model(weight_file_path)
# for i in range(num_output):
#     pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
#     pred[i] = tf.identity(net_model.output, name=pred_node_names[i])
# print('output nodes names are: ', pred_node_names)


# sess = K.get_session()

# if 1:
#     f = 'only_the_graph_def.pb.ascii'
#     tf.train.write_graph(sess.graph.as_graph_def(), export_path, f, as_text=True)
#     print('saved the graph definition in ascii format at: ', osp.join(export_path, f))


# constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
# graph_io.write_graph(constant_graph, export_path, output_graph_name, as_text=False)
# print('saved the constant graph (ready for inference) at: ', osp.join(export_path, output_graph_name))


# for n in constant_graph.node:
#     print(n.name)




# inference model starts here

# if a.inference:
#     num_images_in_validation_set = 60

#     def load_graph(frozen_graph_filename):
#         with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#             graph_def = tf.GraphDef()
#             graph_def.ParseFromString(f.read())
#         with tf.Graph().as_default() as graph:
#             tf.import_graph_def(
#                 graph_def, 
#                 input_map=None, 
#                 return_elements=None, 
#                 name="prefix", 
#                 op_dict=None, 
#                 producer_op_list=None
#             )
#         return graph

#     model = load_graph('./export/constant_graph_weights.pb')


#     # for n in model.as_graph_def().node:
#     #     print n.name


#     # img = plt.imread('./dataset/val/WW/139.jpg')
#     # img = np.reshape(img,[1,256,256,3])
#     # img = img.astype(float)
#     # img = img/255.0
#     # input_node = model.get_tensor_by_name("prefix/conv2d_7_input_2:0")
#     # output_node = model.get_tensor_by_name("prefix/output_node0:0")

#     # with tf.Session(graph=model) as sess:
#     #         # Note: we didn't initialize/restore anything, everything is stored in the graph_def
#     #         y_out = sess.run(output_node, feed_dict={
#     #             input_node: img
#     #         })
#     #         print(y_out)


#     test_data_dir = 'dataset/val'

#     test_generator = test_datagen.flow_from_directory(
#         test_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=1,
#         class_mode='binary',
#         shuffle=False)


#     preds = new_model.predict_generator(generator=test_generator, steps=num_images_in_validation_set)

#     pred_cl = preds > 0.5
#     pred_cl = pred_cl.astype(np.int)
#     pred_for_file = zip(test_generator.filenames, pred_cl, preds)

#     for i in pred_for_file:
#         print(i)
