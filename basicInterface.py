import gym
from collections import defaultdict
import math
import numpy as np
from envs.gridworld_env import CliffWalkingWapper
from Learning.QLearn import QLearning


env = gym.make('CliffWalking-v0', new_step_api=True)
env = CliffWalkingWapper(env)

n_states = env.observation_space.n
m_actions = env.action_space.n 

cfg = {}
agent = QLearning(n_states, m_actions, cfg) # cfg算法参数

def train(cfg, env, agent):
    print("开始训练")
    print(f"环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}")
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0 # 每个回合奖励
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 20 == 0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}, 奖励：{ep_reward:.1f}, Epilon:{agent.epsilon:.2f}, 滑动平均奖励：{ma_rewards[-1]:.1f}")
            



