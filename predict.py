from model import VietOcr
from dataset import DataSet
from generate_dataset import DataGenerator
import tensorflow as tf
import numpy as np


def predict(character_image):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.import_meta_graph('viet_ocr_brain.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    logits = graph.get_tensor_by_name("fc2/logits:0")
    softmax = graph.get_tensor_by_name("softmax:0")

    probs, chars = sess.run([logits, softmax], feed_dict={X: character_image.reshape((1, 28, 28, 1)), keep_prob: 1})

    probs = (np.exp(probs) / np.sum(np.exp(probs))) * 100
    idx = np.argmax(chars)
    return (probs[idx], idx)


ds = DataSet(one_hot=False)
characters = DataGenerator().get_list_characters()

x, y = ds.next_batch_test(1)

print('x.shape', x.shape)
print('y.shape', y.shape)

char, prob = predict(x)

print('Input character: ', characters[int(y[0])])
print('Predicted: ', characters[prob], ' with probability = ', prob)
print('Result: ', characters[int(y[0])] == characters[prob])
print('-' * 10)
