import gym
from envs.gridworld_env import CliffWalkingWapper
env = gym.make('CliffWalking-v0', new_step_api=True)
env = CliffWalkingWapper(env)

n_states = env.observation_space.n
m_actions = env.action_space.n 

print(f"状态数：{n_states}, 动作数：{m_actions}")

