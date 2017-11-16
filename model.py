from dataset import DataSet
from config import *
import tensorflow as tf
import numpy as np
import sys

class VietOcr:
    def __init__(self, weights=None, sess=None, log=True):
        self.sess = sess
        self.log = log
        
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        self.conv2d()
        self.fc_layers()

        self.probs = tf.nn.softmax(self.logits)

        if self.log:
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./log_train', self.sess.graph)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def conv2d(self):

        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([127.5], dtype=tf.float32, shape=[1, 1, 1, 1], name='img_mean')
            images = self.X - mean

        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 1, 32], dtype=tf.float32, stddev=1e-1), 
                                  name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32]), 
                                  dtype=tf.float32, 
                                  trainable=True, 
                                  name='biases')

            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            if self.log:
                tf.summary.histogram('conv1.kernel', kernel)
                tf.summary.histogram('conv1.biases', biases)


        self.pool1 = tf.nn.max_pool(self.conv1_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 name='pool1')

        self.dropout1 = tf.nn.dropout(self.pool1, 
                                    keep_prob=self.keep_prob, 
                                    name='dropout1')

        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.dropout1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            if self.log:
                tf.summary.histogram('conv2.kernel', kernel)
                tf.summary.histogram('conv2.biases', biases)

        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.droupout2 = tf.nn.dropout(self.pool2, keep_prob=self.keep_prob, name='droupout2')

        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.droupout2, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            if self.log:
                tf.summary.histogram('conv3.kernel', kernel)
                tf.summary.histogram('conv3.biases', biases)

        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.dropout3 = tf.nn.dropout(self.pool3, keep_prob=self.keep_prob, name='dropout3')
            
    def fc_layers(self):
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.dropout3.get_shape()[1:]))
            fc1w = tf.get_variable("fc1w", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
            fc1b = tf.Variable(tf.random_normal([625]))

            dropout3_flat = tf.reshape(self.dropout3, [-1, shape])
            fc1 = tf.nn.bias_add(tf.matmul(dropout3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1)
            self.parameters += [fc1w, fc1b]
            if self.log:
                tf.summary.histogram('fc1.weights', fc1w)
                tf.summary.histogram('fc1.biases', fc1b)

        self.dropout_fc1 = tf.nn.dropout(self.fc1, keep_prob=self.keep_prob, name='droupout_fc1')

        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable("fc2w", shape=[625, NO_LABEL], initializer=tf.contrib.layers.xavier_initializer())
            fc2b = tf.Variable(tf.random_normal([NO_LABEL]))

            self.logits = tf.nn.bias_add(tf.matmul(self.dropout_fc1, fc2w), fc2b, name="logits")
            
            self.parameters += [fc2w, fc2b]
            if self.log:
                tf.summary.histogram('fc2.weights', fc2w)
                tf.summary.histogram('fc2.biases', fc2b)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

    def predict(self, character_image):
        char, prob = sess.run([self.logits, self.prod], feed_dict={self.X: character_image})
        
        return (char, prod)


    def train(self, learning_rate, training_epochs, batch_size, keep_prob):

        self.dataset = DataSet()    

        self.Y = tf.placeholder(tf.float32, [None, NO_LABEL], name='Y')
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())

        print('Training...')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(self.dataset.train_idx) / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = self.dataset.next_batch(batch_size)
                feed_dict = { 
                        self.X: batch_xs.reshape([batch_xs.shape[0], 28, 28, 1]), 
                        self.Y: batch_ys, 
                        self.keep_prob: keep_prob
                    }

                summary, c, _ = self.sess.run([self.merged, self.cost, self.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                self.train_writer.add_summary(summary, epoch*i)
            
            if self.log:
                tf.summary.scalar('avg_cost', avg_cost)

            print('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))       

        np.save("vocr.brain", self.parameters)
        print('Training finished!')

    def evaluate(self, batch_size, keep_prob):

        self.Y = tf.placeholder(tf.float32, [None, NO_LABEL], name='Y')

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        N = len(self.dataset.test_idx)
        print('test.size', N);
        correct_sample = 0
        for i in range(0, N, batch_size):
            batch_xs, batch_ys = self.dataset.next_batch_test(batch_size)

            N_batch = batch_xs.shape[0]

            feed_dict = {
                self.X: batch_xs.reshape([N_batch, 28, 28, 1]),
                self.Y: batch_ys,
                self.keep_prob: keep_prob
            }        

            correct = self.sess.run(self.accuracy, feed_dict=feed_dict)            
            correct_sample +=  correct * N_batch

        test_accuracy = correct_sample / N

        print("\nAccuracy Evaluates")
        print("-"*30)
        print('Test Accuracy:', test_accuracy)