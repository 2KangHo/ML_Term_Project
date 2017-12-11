import tensorflow as tf

flag = tf.app.flags


flag.DEFINE_string('train_dir', './model', 'trained model directory')
flag.DEFINE_string('log_dir', './logs', 'log directory')
flag.DEFINE_string('ckpt_name', 'LeeNet.ckpt', 'checkpoint file name')

flag.DEFINE_string('train_data', './input/fashion-mnist_train.csv', 'train data file path')
flag.DEFINE_string('test_data', './input/fashion-mnist_test.csv', 'test data file path')

# hyper parameters
flag.DEFINE_float('l_rate', 0.001, 'learning rate')
flag.DEFINE_integer('b_size', 128, 'mini batch size')
flag.DEFINE_integer('epoch', 32, 'epoch number')


cfg = flag.FLAGS
