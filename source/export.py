from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from keras.models import model_from_config



K.set_learning_phase(0)
config = model.get_config()
weights = model.get_weights()

new_model = Sequential.from_config(config)
new_model.set_weights(weights)

new_model.save('2_classification_test_model.h5')




weight_file_path = './2_classification_test_model.h5'
num_output = 1
prefix_output_node_names_of_final_network = 'output_node'
export_path =  './export' # where to save the exported graph
output_graph_name = 'constant_graph_weights.pb'

K.set_learning_phase(0)
pred = [None]*num_output
pred_node_names = [None]*num_output
net_model = load_model(weight_file_path)
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output, name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)


sess = K.get_session()

if 1:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), export_path, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(export_path, f))




from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, export_path, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(export_path, output_graph_name))



for n in constant_graph.node:
    print(n.name)


