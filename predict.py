from model import VietOcr
from dataset import DataSet
from generate_dataset import DataGenerator

ds = DataSet(one_hot = False)
characters = DataGenerator().get_list_characters()

x, y = ds.get_test_next_batch(1)

print('x.shape', x.shape())
print('y.shape', y.shape())

def __init__(self, weights=None, sess=None,

vocr = VietOcr(weights ='vocr.brain', sess = tf.Session())

char, prob = vocr.predict(x)

print('Test input: ', characters[y[0]])
print('Predict: ', char, ' with probability = ', prob)
print('Result: ', char == characters[y[0]])
print('-'*10)