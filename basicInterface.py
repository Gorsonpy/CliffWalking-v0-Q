import gym
from collections import defaultdict
import math
import numpy as np
from envs.gridworld_env import CliffWalkingWapper
from Learning.QLearn import QLearning
from configs.config import Config, plot_rewards


env = gym.make('CliffWalking-v0', new_step_api=True)
env = CliffWalkingWapper(env)

n_states = env.observation_space.n
m_actions = env.action_space.n 

cfg = Config()
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
            action = agent.sample_action(state)
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
    print("完成训练")
    return {"rewards":rewards, "ma_rewards":ma_rewards}

def test(cfg,env,agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(seed=cfg.seed)  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict_action(state)  # 根据算法选择一个动作
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards":rewards}





