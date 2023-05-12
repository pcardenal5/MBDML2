import numpy as np
import DoubleDQN as ddqn

file_name = '../models/DoubleDQNModel.npy'

trials = np.load(file_name)
ddqn.plot_trials(trials)
ddqn.plot_individual_trial(trials[1])
    
