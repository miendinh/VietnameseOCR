import tensorflow as tf
from model import VietOcr

if __name__ == '__main__':
    sess = tf.Session()

    vocr = VietOcr(weights=None, sess=sess)

    vocr.train(learning_rate=0.001, training_epochs=15, batch_size=1000, keep_prob=0.7)

    vocr.evaluate(batch_size=1000, keep_prob=1)
