import pandas as pd
import numpy as np

class table():
    """"
    actions: the proble action of the env
    epsi: whether to search of use
    learning_rate: the learning fudu
    """
    def __init__(self, actions, epsi,learning_rate = 0.01, reward_decay = 0.9):
        self.epsilon = epsi
        self.lr = learning_rate
        self.gamma = reward_decay
        self.actions = actions
        self.q_table = pd.DataFrame(columns = self.actions, dtype=np.float64)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, obs):
        self.check_state_exist(obs)

        if np.random.uniform() > self.epsilon:

            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table.loc[obs, : ]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()

        return action

    def learning(self, state, action, reward, s_):
        self.check_state_exist(state)
        self.check_state_exist(s_)
        q_pred = self.q_table.loc[state, action]

        if s_ is not 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, : ].max()
        else:
            q_target = reward

        self.q_table.loc[state, action] += self.lr * (q_target - q_pred)