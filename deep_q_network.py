#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
tf = tf.compat.v1
tf.disable_eager_execution()


import tensorflow as tf
tf = tf.compat.v1
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # 游戏名称，用于日志文件
ACTIONS = 2 # 有效动作的数量
GAMMA = 0.99 # 过去观察的衰减率
OBSERVE = 100000. # 训练之前观察的时间步数
EXPLORE = 2000000. # epsilon退火的帧数
FINAL_EPSILON = 0.0001 # epsilon的最终值
INITIAL_EPSILON = 0.0001 # epsilon的初始值
REPLAY_MEMORY = 50000 # 记忆的先前转换数量
BATCH = 32 # minibatch的大小
FRAME_PER_ACTION = 1
job_prefix = '0803/'

def weight_variable(shape):
    """
    创建权重变量的函数

    参数：
    - shape：权重的形状

    返回：
    - 权重变量
    """
    initial = tf.random.truncated_normal(shape, stddev = 0.01)
    # 生成截断的正态分布随机数
    # 以均值为0，标准差为0.01生成随机数
    return tf.Variable(initial)

def bias_variable(shape):
    """
    创建偏置变量的函数

    参数：
    - shape：偏置的形状

    返回：
    - 偏置变量
    """
    initial = tf.constant(0.01, shape = shape)
    # 生成常数张量
    # 值为0.01，形状为shape
    return tf.Variable(initial)

def conv2d(x, W, stride):
    """
    创建卷积层的函数

    参数：
    - x：输入张量
    - W：卷积核权重
    - stride：卷积步长

    返回：
    - 卷积结果张量
    """
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    """
    创建最大池化层的函数

    参数：
    - x：输入张量

    返回：
    - 池化结果张量
    """
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    """
    创建深度Q网络的函数

    返回：
    - 输入张量s
    - 输出张量readout
    - 隐藏层张量h_fc1
    """
    # 网络权重
    W_conv1 = weight_variable([8, 8, 4, 32])
    # conv的全称是convolution，表示卷积层
    # 卷积层的作用是提取图像的特征，通过卷积核与输入图像进行卷积操作，得到特征图
    # 这个卷积层有8x8的卷积核，4个输入通道，32个输出通道
    b_conv1 = bias_variable([32])
    # 偏置用来“加在某个输出上”，用于调整神经元的输出值

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    # fc 的全称是fully connected，表示全连接层
    b_fc2 = bias_variable([ACTIONS])

    # 输入层
    s = tf.placeholder("float", [None, 80, 80, 4])
    # placeholder 是占位符的意思，用于定义过程输入数据的位置

    # 隐藏层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    # relu是激活函数，用于增加网络的非线性，将负值转换为0
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    """
    训练深度Q网络的函数

    参数：
    - s：输入张量
    - readout：输出张量
    - h_fc1：隐藏层张量
    - sess：TensorFlow会话对象
    """
    # 定义损失函数
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1) # 乘法
    cost = tf.reduce_mean(tf.square(y - readout_action)) # 平方差
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost) # Adam优化器 AdamW

    # 打开游戏状态以与模拟器通信
    game_state = game.GameState()

    # 在回放内存中存储先前的观察
    D = deque()

    # 打印信息
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 通过不做任何操作获取第一个状态，并将图像预处理为80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 保存和加载网络
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    # checkpoint = tf.train.get_checkpoint_state("saved_networks")
    # checkpoint = None
    checkpoint = tf.train.get_checkpoint_state(job_prefix + 'saved_networks/')
    if checkpoint and checkpoint.model_checkpoint_path: # if checkpoint is not None
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird": # while True:
        # epsilon贪婪地选择一个动作
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # 什么都不做

        # 缩小epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作并观察下一个状态和奖励
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t) # 获取游戏帧
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY) # 转换为灰度
        # 转换为灰度就是将图像转换为黑白图像（只有黑白和不同程度的灰），而二值化时只有黑色和白色，没有其他颜色
        # 灰度：0-255，二值化：0或255
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY) # 二值化
        x_t1 = np.reshape(x_t1, (80, 80, 1)) # 重塑形状为80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2) # 将新帧添加到状态中

        # 将转换存储在D中
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 只有在观察完成后才进行训练
        if t > OBSERVE:
            # 从回放内存中随机采样一个minibatch进行训练
            minibatch = random.sample(D, BATCH)

            # 获取批量变量
            s_j_batch = [d[0] for d in minibatch] # 当前状态
            a_batch = [d[1] for d in minibatch] # 动作
            r_batch = [d[2] for d in minibatch] # 奖励
            s_j1_batch = [d[3] for d in minibatch] # 下一个状态

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 如果是终止状态，只等于奖励
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 执行梯度下降步骤
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # 更新旧值
        s_t = s_t1
        t += 1

        # 每10000次迭代保存进度
        if t % 10000 == 0:
            saver.save(sess, job_prefix + 'saved_networks/' + GAME + '-dqn', global_step = t)

        # 打印信息
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))

def playGame():
    # 配置TensorFlow会话
    # TensorFlow会话是TensorFlow操作的执行环境，它封装了一些操作的执行状态和操作的执行设备
    sess = tf.InteractiveSession()
    # 创建网络
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
