import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    def __init__(self, env, epsilon, learning_rate):
        self.gamma = GAMMA
        self.lr = learning_rate
        self.replay_size = REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.epsilon = epsilon    #e_greedy

        self.replay = deque()

        self.n_state = env.observation_space.shape[0]   #the dim of the state
        self.n_action = env.action_space.n    #the dim of the action

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_state])   #the input of the network

        self.action_input = tf.placeholder(tf.float32, [None, self.n_action])
        self.y_input = tf.placeholder(tf.float32, [None])

        self._build_network(25)
        self._bulid_optimizer()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # build the network
    # h_size : the size of the hidden layer
    def _build_network(self, h_size):

        Weight = {
            'w1' : tf.Variable(tf.truncated_normal([self.n_state, h_size]),dtype=tf.float32),
            'w2' : tf.Variable(tf.truncated_normal([h_size, self.n_action]), dtype=tf.float32)
        }

        Biases = {
            'b1' : tf.Variable(tf.zeros([h_size])),
            'b2' : tf.Variable(tf.zeros([self.n_action]))
        }

        out1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, Weight['w1']), Biases['b1']))
        self.q_value = tf.add(tf.matmul(out1, Weight['w2']), Biases['b2'])

    def _bulid_optimizer(self):
        Q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def choose_action(self, state):

        if np.random.uniform() > self.epsilon:
            return random.randint(0, self.n_action - 1)
        else:
            value = self.sess.run(self.q_value[0],feed_dict = {
                    self.state_input : [state]
                })
            return np.argmax(value)


    def process(self, state, action, reward, state_):
        action_oh = np.zeros(self.n_action, dtype=np.int32)
        action_oh[action] = 1

        self.replay.append((state, action_oh, reward, state_))

        if len(self.replay) > self.replay_size:
            self.replay.popleft()

        if len(self.replay) > self.batch_size:
            self.train_network()

    def train_network(self):

        batch = random.sample(self.replay, self.batch_size)
        state_batch = [d[0] for d in batch]
        action_batch = [d[1] for d in batch]
        reward_batch = [d[2] for d in batch]
        next_state_batch = [d[3] for d in batch]

        y_batch = []
        Q_value_batch = self.sess.run(self.q_value, feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.sess.run(self.optimizer,feed_dict={
                self.state_input : state_batch,
                self.action_input : action_batch,
                self.y_input : y_batch
            })

# ---------------------------------------------------------
# Hyper Parameters for the running env
ENV_NAME = 'CartPole-v0'
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():

    env = gym.make(ENV_NAME)
    env = env.unwrapped

    agent = DQN(env, epsilon=0.9, learning_rate=0.1)

    while 'cartpole' is not 'cp':

        state = env.reset()
        env.render()
        for step in range(STEP):
            action = agent.choose_action(state)

            state_, reward, done, _ = env.step(action)

            agent.process(state, action, reward, state_)

            if done:
                break

if __name__ == '__main__':
    main()