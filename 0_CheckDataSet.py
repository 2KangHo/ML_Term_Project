import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from config import cfg

CLASSES = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}

train_data = pd.read_csv(cfg.train_data)
x_train = train_data.iloc[:,1:].values
y_train = train_data.iloc[:,:1].values
test_data = pd.read_csv(cfg.test_data)
x_test = test_data.iloc[:,1:].values
y_test = test_data.iloc[:,:1].values

print(train_data.shape, x_train.shape, y_train.shape)
print(test_data.shape, x_test.shape, y_test.shape)

print("Label: ", y_train[123], '-', CLASSES[str(y_train[123][0])])
plt.imshow(x_train[123,:].reshape(28,28), cmap='gray', interpolation='nearest')
plt.show()
print("Label: ", y_test[7], '-', CLASSES[str(y_test[7][0])])
plt.imshow(x_test[7,:].reshape(28,28), cmap='gray', interpolation='nearest')
plt.show()