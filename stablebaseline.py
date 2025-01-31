from environment import *
from sb3_contrib import RecurrentPPO

env = HarlowEnv()
env = MetaLearningWrapper(env)
    

model = RecurrentPPO(
    policy = 'MlpLstmPolicy',
    env = env,
    verbose = 1,
    learning_rate = 7e-4,
    n_steps = 40,
    gamma = 1.0,
    ent_coef = 0.05,
)

model.learn(total_timesteps = 1000000)