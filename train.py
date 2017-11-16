import tensorflow as tf
from model import VietOcr

if __name__ == '__main__':

    sess = tf.Session()

    vocr = VietOcr(weights = None, sess = sess)

    vocr.train(learning_rate = 10, training_epochs = 1, batch_size = 100, keep_prob = 0.7)

    vocr.evaluate(batch_size = 100, keep_prob = 1)