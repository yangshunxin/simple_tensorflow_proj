import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_NODE = 784 #输入像素点个数
OUTPUT_NODE = 10 #输出分类个数

# 配置神经网络的参数
LAYER1_NODE = 500 #隐藏层节点个数

BATCH_SIZE = 100 #一个batch输入样本的个数

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减系数

REGULARIZATION_RATE = 0.0001 # L2正则项的系数
TRAIN_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减

# 获取前向传播的结果

def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    # 没有滑动平滑模型，直接用参数取当前的值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        #计算输出层结果，不做softmax，因为后面函数会有该计算过程
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用avg_class.average函数来计算得出变量的滑动平均值
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2))

#模型训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='x-output')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不使用滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 保存训练轮数的变量，不可训练
    global_step = tf.Variable(0, trainable=False)

    # 用滑动平均衰减率和训练轮数 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，其它变量不需要了， tf.trainable_variables
    # Graphkeys.TRAINABLE_VARIABLES 中的元素，其中的变量没有指定 trainable=False
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算滑动平均后的前向传播的结果，内部有一个影子变量；
    # 用于计算验证集上的准确率
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵 共有两个函数，记住哟
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))

    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2正则损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总的损失
    loss = cross_entropy_mean + regularization

    # 设置衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础学习率
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 使用梯度下降优化函数----这时正常训练的终极操作，，BP过程，会改变weight和bias的值
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练网络模型时，每过一遍数据需要反向传播更新网络中的参数，
    # 然后更新每一个参数的滑动平均值，通过tf.control_dependencies机制
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向结果，是否正确，
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证数据集
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 测试数据集
        test_feed = {x: mnist.test.images, y_:mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAIN_STEPS):
            # 每1000轮 输出一次验证集上的测试数据
            if i % 1000 == 0:
                # 计算滑动验证集上的结果，
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("after %d training steps, useing average model validation accuray is %g"%(i, validate_acc))

            # 产生这一轮使用的batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束后， 在测试数据上检验神经网络的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step, test accuracy using average model is %g"%(TRAIN_STEPS, test_acc))

#主程序入口
def main(argv=None):
    minst = input_data.read_data_sets("./tmp/data", one_hot=True)
    train(mnist=minst)

# tf.app.run() 会调用main函数
if __name__ == '__main__':
    tf.app.run()