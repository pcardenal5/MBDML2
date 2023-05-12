import DoubleDQN as ddqn
import numpy as np


parameterDictionary = {
# CARTPOLE GAME SETTINGS    
    'OBSERVATION_SPACE_DIMS' : 6,
    'ACTION_SPACE' : [0,1,2],
    
# AGENT/NETWORK HYPERPARAMETERS
    'EPSILON_INITIAL': 0.5,             # exploration rate
    'EPSILON_DECAY': 0.99,
    'EPSILON_MIN': 0.01,
    'ALPHA': 0.001,                     # learning rate
    'GAMMA': 0.99,                      # discount factor
    'TAU': 0.1,                         # target network soft update hyperparameter
    'HIDDEN_LAYER_SIZE' : 64,
    'EXPERIENCE_REPLAY_BATCH_SIZE': 32,
    'AGENT_MEMORY_LIMIT': 200,
    'MIN_MEMORY_FOR_EXPERIENCE_REPLAY': 100,
    
#Agent test hyperparameters
    'NUMBER_OF_TRIALS' : 1,
    'MAX_TRAINING_EPISODES' :  10,
    'MAX_STEPS_PER_EPISODE' :  10,
}

trials , agent = ddqn.test_agent(parameterDictionary)



file_name = 'DoubleDQNModel.npy'



print ('Saving', file_name)
np.save(file_name, trials)
