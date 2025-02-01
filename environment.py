import numpy as np
import random

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete


class HarlowEnv(gym.Env):
    """
    A Harlow environment.
    """

    def __init__(
            self,
            num_trials = 20,
            flip_prob = 0.2, 
            seed = None,
        ):

        """
        Construct an environment.
        """

        self.num_trials = num_trials # max number of trials per episode
        self.flip_prob = flip_prob # flip probability

        # set random seed
        self.set_random_seed(seed)

        # initialize action and observation spaces
        self.action_space = Discrete(3) # action 0, 1, 2 represent choosing left, choosing right, fixating, respectively
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = (1,))


    def reset(self):
        """
        Reset the environment.
        """

        # reset the environment
        self.num_completed = 0 # number of trials completed
        self.stage = 'fixation' # set the initial stage to fixation
        self.correct_answer = np.random.randint(0, 2) # randomly pick a correct answer

        obs = np.array([0.]) # use 0 to represent fixation stage
        
        info = {
            'correct_answer': self.correct_answer, # include the correct answer in the information
            'mask': self.get_action_mask()
        }

        return obs, info
    

    def step(self, action):
        """
        Step the environment.
        """

        done = False # initialize to False

        # fixation stage
        if self.stage == 'fixation':
            # fixation action
            if action == 2:
                reward = 0.
            
            # decision action
            else:
                reward = -1. # give a panelty if the agent doesn't execute the fixation action
                
            self.stage = 'decision' # set the stage to decision
            
            obs = np.array([1.]) # use 1 to represent decision stage 
        
        # decision stage
        elif self.stage == 'decision':
            self.num_completed += 1 # update number of completed trials
            self.flip_bandit() # randomly flip the bandit

            if action == self.correct_answer:
                reward = 1.
            else:
                reward = -1.
            
            self.stage = 'fixation' # set the stage to fixation
            
            obs = np.array([0.])
        
        # end the episode
        if self.num_completed >= self.num_trials:
            done = True
        
        info = {
            'correct_answer': self.correct_answer,
            'mask': self.get_action_mask(),
        }

        return obs, reward, done, False, info
    

    def flip_bandit(self):
        """
        Flip the bandit.
        """

        if np.random.random() < self.flip_prob:
            self.correct_answer = 1 - self.correct_answer

    
    def get_action_mask(self):
        """
        Get action mask.

        Valid actions are set to True while invalid actions are set to False
        """

        mask = np.ones((self.action_space.n,), dtype = bool)

        return mask
    

    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    
    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        self.init_prev_variables()

        new_observation_shape = (
            self.env.observation_space.shape[0] +
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        obs_wrapped = self.wrap_obs(obs)

        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        self.init_prev_variables()

        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset action and reward from the previous trial.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with action and reward from the previous trial.
        """

        obs_wrapped = np.hstack([
            obs,
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward
        ])
        return obs_wrapped



if __name__ == '__main__':
    # testing
    
    env = HarlowEnv()
    env = MetaLearningWrapper(env)
    
    for i in range(50):

        obs, info = env.reset()
        print('initial obs:', obs)
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(
                'correct answer:', info['correct_answer'], '|',
                'action:', action, '|',
                'reward:', reward, '|',
                'next obs:', obs, '|',
                'done:', done, '|',
            )
