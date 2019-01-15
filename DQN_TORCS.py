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
# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 50 # experience replay buffer size
BATCH_SIZE = 5 # size of minibatch


# The DQN class
class DQN():
  # DQN Agent
  def __init__(self):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = 29  # GYM_TORCS
    self.action_dim = 3  #steering_left $ acceleration & brake 

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.initialize_all_variables())


  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done): #感知信息
    one_hot_action = np.zeros(self.action_dim)
    print("one_hot_action")
    print(one_hot_action)
    one_hot_action=action####################################################################

    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()
    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

##################################################################################################
  def egreedy_action(self,state): #获取包含随机的动作
    Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
    #return Q_value
    if random.random() <= self.epsilon:
      return (random.randint(0,self.action_dim - 1),random.randint(0,self.action_dim - 1) ,random.randint(0,self.action_dim - 1)  )

    else:
      return Q_value

    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000  ###############################################

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

# end of the DQN class


# Hyper Parameters
EPISODE = 10000 # Episode limitation
STEP = 30 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

# the main loop
def playGame():
        # Generate a Torcs environment
        env = TorcsEnv(vision=False, throttle=True,gear_change=False)
        agent = DQN()

        for episode in range(EPISODE):
            if np.mod(episode, 3) == 0:
              ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
            else:
              ob = env.reset()

            state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))     

          # Train
            for step in range(STEP):
              action = agent.egreedy_action(state) # e-greedy action for train########################################3


              ob, r_t, done, info = env.step(action)
              state_next = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))        


              agent.perceive(state,action,r_t,state_next,done)
              state = state_next

              if done:
               break

        env.end()  # This is for shutting down TORCS
        print("Finish.")




if __name__ == '__main__':
    playGame()