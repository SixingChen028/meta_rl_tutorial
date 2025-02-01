from environment import *
from sb3_contrib import RecurrentPPO

env = HarlowEnv()
env = MetaLearningWrapper(env)
    

model = RecurrentPPO(
    policy = 'MlpLstmPolicy',
    env = env,
    verbose = 1,
    learning_rate = 5e-4,
    n_steps = 40,
    gae_lambda = 1,
    n_epochs = 1,
    batch_size = 16,
    gamma = 1.0,
    vf_coef = 0.05,
    ent_coef = 0.05,
    policy_kwargs = dict(lstm_hidden_size = 32)
)

model.learn(total_timesteps = 1000000)