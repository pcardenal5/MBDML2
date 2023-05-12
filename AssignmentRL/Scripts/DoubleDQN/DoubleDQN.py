import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from matplotlib import pyplot as plt


def create_dqn(parameterDictionary):
    # not actually that deep
    hls = parameterDictionary['HIDDEN_LAYER_SIZE']
    nn = Sequential()
    nn.add(Dense(hls, input_dim=parameterDictionary['OBSERVATION_SPACE_DIMS'], activation='relu'))
    nn.add(Dense(hls, activation='relu'))
    nn.add(Dense(len(parameterDictionary['ACTION_SPACE']), activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=parameterDictionary['ALPHA']))
    return nn


class DoubleDQNAgent(object):

       
    def __init__(self, parameterDictionary):
        self.memory = []
        self.online_network = create_dqn(parameterDictionary)
        self.target_network = create_dqn(parameterDictionary)
        self.epsilon = parameterDictionary['EPSILON_INITIAL']
        self.pD = parameterDictionary
        self.has_talked = False
    
    
    def act(self, state):
        if self.epsilon > np.random.rand():
            # explore
            return np.random.choice(self.pD['ACTION_SPACE'])
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.online_network.predict(state)[0]
            return np.argmax(q_values)


    def experience_replay(self):

        minibatch = random.sample(self.memory, self.pD['EXPERIENCE_REPLAY_BATCH_SIZE'])
        minibatch_new_q_values = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.online_network.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(self.online_network.predict(next_state))
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network.predict(next_state)[0][online_net_selected_action]
                q_update = reward + self.pD['GAMMA'] * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.online_network.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)
        
        
    def update_target_network(self):
        q_network_theta = self.online_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1-self.pD['TAU']) + q_weight * self.pD['TAU']
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)


    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) <= self.pD['AGENT_MEMORY_LIMIT']:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)
                  
                  
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.pD['EPSILON_DECAY'], self.pD['EPSILON_MIN'])


    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, self.pD['OBSERVATION_SPACE_DIMS']))  

    def get_networks(self):
        return self.online_network, self.target_network

def test_agent(parameterDictionary):
    env = gym.make('Acrobot-v1')
    env.seed(42)
    trials = []
    NUMBER_OF_TRIALS = parameterDictionary['NUMBER_OF_TRIALS']
    MAX_TRAINING_EPISODES = parameterDictionary['MAX_TRAINING_EPISODES']
    MAX_STEPS_PER_EPISODE = parameterDictionary['MAX_STEPS_PER_EPISODE']

    for trial_index in range(NUMBER_OF_TRIALS):
        agent = DoubleDQNAgent(parameterDictionary)
        trial_episode_scores = []

        for episode_index in range(1, MAX_TRAINING_EPISODES+1):
            state = env.reset()
            episode_score = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                episode_score += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > agent.pD['MIN_MEMORY_FOR_EXPERIENCE_REPLAY']:
                    agent.experience_replay()
                    agent.update_target_network()
                if done:
                    break
            
            trial_episode_scores.append(episode_score)
            agent.update_epsilon()
            last_100_avg = np.mean(trial_episode_scores[-100:])
            print ("Episode %d finished. Scored %d, avg %.2f" % (episode_index, episode_score, last_100_avg))
        trials.append(np.array(trial_episode_scores))
    return np.array(trials), agent

def plot_trials(trials):
    _, axis = plt.subplots()    

    for i, trial in enumerate(trials):
        steps_till_solve = trial.shape[0]-100
        # stop trials at 2000 steps
        if steps_till_solve < 1900:
            bar_color = 'b'
            bar_label = steps_till_solve
        else:
            bar_color = 'r'
            bar_label = 'Stopped at 2000'
        plt.bar(np.arange(i,i+1), steps_till_solve, 0.5, color=bar_color, align='center', alpha=0.5)
        axis.text(i-.25, steps_till_solve + 20, bar_label, color=bar_color)

    plt.ylabel('Episodes Till Solve')
    plt.xlabel('Trial')
    trial_labels = [str(i+1) for i in range(len(trials))]
    plt.xticks(np.arange(len(trials)), trial_labels)
    # remove y axis labels and ticks
    axis.yaxis.set_major_formatter(plt.NullFormatter())
    plt.tick_params(axis='both', left='off')

    plt.title('Double DQN CartPole v-0 Trials')
    plt.show()


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN CartPole v-0 Steps in Select Trial')
    plt.show()
