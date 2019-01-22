import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
# RF imports
from gym_torcs import TorcsEnv
import gym
import tensorflow as tf
import random
from collections import deque
import cv2

# Hyper Parameters for DQN
ACTIONS = 3 				# number of valid actions
GAMMA = 0.99 				# decay rate of past observations
OBSERVE = 10 			# timesteps to observe before training
EXPLORE = 20. 		    # frames over which to anneal epsilon

REPLAY_MEMORY = 50 		# number of previous transitions to remember
BATCH = 5 					# size of minibatch
##########################################################################################################################################################
# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01) #tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
    return tf.Variable(initial)#
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# 卷积函数
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME") 
    #实现卷积的函数
# 池化 核 2*2 步长2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
# CNN
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])    # 卷积核patch的大小是8*8, RGBD,channel是4,输出是32个featuremap
    b_conv1 = bias_variable([32])				# 传入它的shape为[32]

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    # input layer 输入层 输入向量为80*80*4
    s = tf.placeholder("float", [None, 80, 80, 4])					# 
    #print("s.shape",s.shape)
    # 第一个隐藏层+一个池化层
    h_conv1 = tf.nn.tanh(conv2d(s, W_conv1, 4) + b_conv1)			#  
    h_pool1 = max_pool_2x2(h_conv1)									# 
    #print("h_pool1.shape",h_pool1.shape)
    #第二个隐藏层
    h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2, 2) + b_conv2)		#   
    # 第三个隐藏层
    h_conv3 = tf.nn.tanh(conv2d(h_conv2, W_conv3, 1) + b_conv3)		# 
    #print("h_conv3.shape",h_conv3.shape)
    #展平
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    #print("h_conv3_flat.shape",h_conv3_flat.shape)
    # 第一个全连接层
    h_fc1 = tf.nn.tanh(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    #print("h_fc1.shape",h_fc1.shape)
    # readout layer  输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    #print("readout.size",readout.shape)
    return s, readout, h_fc1
###################################################################################################################################################

# Hyper Parameters
EPISODE = 10000 # Episode limitation
STEP = 20 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function  定义损失函数
    a = tf.placeholder("float", [None, ACTIONS]) #tf.placeholder 是 Tensorflow 中的占位符，暂时储存变量
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1) #矩阵按行求和,multiply这个函数实现的是元素级别的相乘
    cost = tf.reduce_mean( tf.square(y - readout_action) ) #张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # store the previous observations in replay memory
    D = deque()
    # start training
    epsilon = 0.1
    for episode in range(EPISODE):
        # open up a game state to communicate with emulator
        env = TorcsEnv(vision=True, throttle=True,gear_change=False)
        if np.mod(episode, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        if episode==0:
            # get the first state by doing nothing and preprocess the image to 80x80x4
            do_nothing = np.zeros(ACTIONS)
            ob, r_0, terminal, info = env.step(do_nothing)
            x_t=ob.img
            print("type(x_t)",type(x_t))
            #x_t, r_0, terminal = game_state.frame_step(do_nothing)
            x_t=x_t.swapaxes(0,2)
            print("x_t.shape",x_t.shape)
            #将图像转换成80*80，并进行灰度化
            #Resize image to 80x80, Convert image to grayscale,remove the background appeared in the original game can make it converge faster
            x_t=cv2.resize(x_t, (80, 80))
            print("x_t.shape",x_t.shape)
            x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)  
            ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)  #对图像进行二值化,从灰度图像中获取二进制图像或用于消除噪声，即滤除太小或太小的像素
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 将图像处理成4通道,stack last 4 frames to produce an 80x80x4 input array for network

        for step in range(STEP):
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]   #将当前环境输入到CNN网络中
            print("readout_t",readout_t)
            a_t = np.zeros([ACTIONS])
            if random.random() <= epsilon:
                print("Random Action")
                a_t[0] = random.random()
                a_t[1] = random.random()
                a_t[2] = random.random()
            else:
                a_t[0] = readout_t[0]
                a_t[1] = readout_t[1]
                a_t[2] = readout_t[2]
            # run the selected action and observe next state and reward
            ob, r_t, terminal, info = env.step(a_t)
            x_t1_colored=ob.img
            x_t1_colored=x_t1_colored.swapaxes(0,2)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            # store the transition in D,经验池保存的是以一个马尔科夫序列于D中
            D.append((s_t, a_t, r_t, s_t1, terminal))
        
            if len(D) > REPLAY_MEMORY:
                D.popleft()
            # only train if done observing
            if step > OBSERVE:  # timesteps to observe before training
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                y_batch = []  #y_batch表示标签值，如果下一时刻游戏关闭则直接用奖励做标签值，若游戏没有关闭，则要在奖励的基础上加上GAMMA比例的下一时刻最大的模型预测值

                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                # perform gradient step
                train_step.run(  feed_dict = {y : y_batch, a : a_batch, s : s_j_batch}  )

            # update the old values
            s_t = s_t1
            step += 1
            # save progress every 10000 iterations
            if step % 10000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = step)

            # print info
            state = ""
            if step <= OBSERVE:
                state = "observe"
            elif step > OBSERVE and step <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", step, "/ STATE", state,  "/ EPSILON", epsilon, "/ REWARD", r_t,  "/ Q_MAX %e" % np.max(readout_t))




def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    sess.run(tf.initialize_all_variables())
    print("create net work successfully")
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == '__main__':
    playGame()