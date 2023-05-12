import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras import Model

from IPython.display import clear_output
from tensorflow.keras.models import load_model


class ReplayBuffer():
    def __init__(self, maxsize, statedim, naction):
        self.cnt = 0
        self.maxsize = maxsize
        self.state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
        self.action_memory = np.zeros((maxsize, naction), dtype=np.float32)
        self.reward_memory = np.zeros((maxsize,), dtype=np.float32)
        self.next_state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
        self.done_memory = np.zeros((maxsize,), dtype= np.bool)

    def storexp(self, state, action, reward, next_state, done):
        index = self.cnt % self.maxsize
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = 1- int(done)
        self.cnt += 1

    def sample(self, batch_size):
        max_mem = min(self.cnt, self.maxsize)
        batch = np.random.choice(max_mem, batch_size, replace= False)  
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.done_memory[batch]
        return states, next_states, rewards, actions, dones


class Actor(Model):
    def __init__(
            self, 
            state_size: int, 
            action_size: int,
            hidden_size: int):
        
        """Initialization."""
        
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        # set the hidden layers
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.policy =  tf.keras.layers.Dense(action_size, activation='tanh')

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        policy = self.policy(layer2)
        return policy
    
    
class CriticQ(Model):
    def __init__(
        self, 
        hidden_size : int
    ):
        """Initialize."""
        super(CriticQ, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation = None)

    def call(self, state, action):
        layer1 = self.layer1(tf.concat([state, action], axis=1))
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value
    
    
    
class Agent():
    """
        
    Attributes:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (tf.keras.Model): target actor model to select actions
        critic (tf.keras.Model): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
        hidden_size: int
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            gamma (float): discount factor
        """
        
        # CREATING THE Q-Network
        self.env = env
        
        self.state_size = self.env.observation_space.shape
        self.action_size = env.action_space.n
        self.hidden_size = hidden_size
        
        self.actor_lr = 7e-3
        self.critic_lr = 7e-3
        self.gamma = 0.99    # discount rate
        self.actor = Actor(self.state_size, self.action_size, self.hidden_size)
        self.actor_target = Actor(self.state_size, self.action_size,hidden_size)
        self.critic1 = CriticQ(self.hidden_size)
        self.critic2 = CriticQ(self.hidden_size)
        self.critic_target1 = CriticQ(self.hidden_size)
        self.critic_target2 = CriticQ(self.hidden_size)
        self.batch_size = 64
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt1 = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt2 = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.c_opt1 = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.c_opt2 = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.memory = ReplayBuffer(1_00_000, env.observation_space.shape, env.action_space.n)
        self.trainstep = 0
        self.update_freq = 5
        self.min_action = 0
        self.max_action = 2
        self.actor_update_steps = 2
        self.warmup = 200
        self.update_target()
    
    def get_action(self, state, evaluate=False):
        if self.trainstep > self.warmup:
            evaluate = True
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=[self.action_size], mean=0.0, stddev=0.1)

        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
        print(actions)
        return actions[0]
    
    
    def savexp(self,state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

    def update_target(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target1.set_weights(self.critic1.get_weights())
        self.critic_target2.set_weights(self.critic2.get_weights())

    def train_step(self):
        if self.memory.cnt < self.batch_size:
            return 

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states      = tf.convert_to_tensor(states, dtype= tf.float32)
        actions     = tf.convert_to_tensor(actions, dtype= tf.float32)
        rewards     = tf.convert_to_tensor(rewards, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
        # dones       = tf.convert_to_tensor(dones, dtype= tf.bool)
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            curr_Q1s = tf.squeeze(self.critic1(states, actions), 1)
            curr_Q2s = tf.squeeze(self.critic2(states, actions), 1)

            next_P_targs = self.actor_target(next_states)
            next_P_targs += tf.clip_by_value(tf.random.normal(shape=[*np.shape(next_P_targs)], mean=0.0, stddev=0.2), -0.5, 0.5)
            next_P_targs = self.max_action * (tf.clip_by_value(next_P_targs, self.min_action, self.max_action))

            next_Q_targs = tf.squeeze(self.critic_target1(next_states, next_P_targs), 1)
            next_Q2_targs  = tf.squeeze(self.critic_target2(next_states, next_P_targs), 1)
            next_target_Qs = tf.math.minimum(next_Q_targs, next_Q2_targs)
            expected_Qs    = rewards + self.gamma * next_target_Qs * dones
            
            critic_loss1 = tf.keras.losses.MSE(expected_Qs, curr_Q1s)
            critic_loss2 = tf.keras.losses.MSE(expected_Qs, curr_Q2s)
          
        grads1 = tape1.gradient(critic_loss1, self.critic1.trainable_variables)
        grads2 = tape2.gradient(critic_loss2, self.critic2.trainable_variables)
      
        self.c_opt1.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grads2, self.critic2.trainable_variables))
        
        self.trainstep +=1
        
        if self.trainstep % self.actor_update_steps == 0:
            with tf.GradientTape() as tape3:
                
                curr_Ps = self.actor(states)
                actor_loss = -self.critic1(states, curr_Ps)
                actor_loss = tf.math.reduce_mean(actor_loss)
          
            grads3 = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.a_opt.apply_gradients(zip(grads3, self.actor.trainable_variables))

        if self.trainstep % self.update_freq == 0:
            self.update_target()
