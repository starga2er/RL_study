import gym, ray, os
from ray.rllib.agents import ppo

# env = gym.make('CartPole-v0')

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('CartPole-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        self.env.render()
        return self.env.step(action)

ray.init()

if __name__ == "__main__":
    
    agent = ppo.PPOTrainer(env= MyEnv , config={
    "framework": "torch",
    "env_config": {},  # config to pass to env class
    "num_gpus": 1,
    "num_workers": 1,
    })

    
    N=10
    results = []
    episode_data = []
    for n in range(N):
        result = agent.train()
        results.append(result)
        episode = {'n': n, 
                   'episode_reward_min':  result['episode_reward_min'],  
                   'episode_reward_mean': result['episode_reward_mean'], 
                   'episode_reward_max':  result['episode_reward_max'],  
                   'episode_len_mean':    result['episode_len_mean']}    
        episode_data.append(episode)
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')

    print(agent.save(os.getcwd() + "/save_data/my_tutorial_data"))
