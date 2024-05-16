import numpy as np
import math
from collections import defaultdict

class QLearning:
    def __init__(self, n_states, n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def choose_action(self, state):
        self.sample_count += 1
        # 采用epsilon-greedy策略选择动作 指数递减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.sample_count / self.epsilon_decay)
        # 带有探索的贪心策略 区间[0, 1] 之间的随机数小于epsilon则选择随机动作
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选择Q(s,a)最大的动作
        else:
            action = np.random.choice(self.action_dim) # 随机选择动作
        return action

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)