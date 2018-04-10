import pandas as pd
import numpy as np

class robot():

    def __init__(self, actions, lr = 0.8, e_greedy = 0.96, namda = 0.8, userobot = True):
        self.actions = actions
        self.lr = lr
        self.epsilon = e_greedy
        self.namda = namda
        self.userobot = userobot
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)

    def choose_action(self, obs):
        self.is_state_exist(obs)

        if self.userobot:
            if np.random.uniform() > self.epsilon:
                action = np.random.choice(self.actions)

            else:
                state_action = self.q_table.loc[obs, : ]
                state_action = state_action.reindex(np.random.permutation(state_action.index))
                action = state_action.idxmax()

        else:
            action = np.random.choice(self.actions)

        return action


    def learning(self, obs, action, reward, obs_):
        self.is_state_exist(obs)
        self.is_state_exist(obs_)

        q_value = self.q_table.loc[obs, action]

        next_action = self.choose_action(obs_)
        q_target = reward + self.namda * self.q_table.loc[obs_, next_action]

        self.q_table.loc[obs, action] += self.lr * (q_target - q_value)


    def is_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index = self.actions,
                    name = state
                )
            )
