import os
import pickle
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

from constants import ROOT_DIR
from priority_buffer import PriorityExperienceReplay
from models import DNN, UserMovieEmbedding, StateRepresentation


class DdqnAgent(object):
    def __init__(self, env, users_num, items_num, state_size, action_size, epsilon=0.8, epsilon_min=0.01, epsilon_decay=0.999, embedding_dim=100, hidden_dim=256, learning_rate=0.001, gamma = 0.8, buffer_size=5000,batch_size=64):
        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.state_size = state_size
        self.action_size = action_size

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
     
        self.batch_size = batch_size
        
        # Experience replay buffer
        # self.replay_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer = PriorityExperienceReplay(self.buffer_size, self.embedding_dim)

        #epsilon of 0.8 denotes we get 20% random decision
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q network and target network initialization
        self.q_network = DNN(self.embedding_dim, self.hidden_dim, self.action_size)
        self.target_network = DNN(self.embedding_dim, self.hidden_dim, self.action_size)
        
        # Optimizer initialization
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        # Embedding layer
        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)),np.zeros((1,))])
        
        # Save model in the path below
        self.save_model_weight_dir = f"{ROOT_DIR}/rl_model/ddqn/trail-{datetime.now().strftime('%Y-%m-%d-%H')}"
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'images'))
        embedding_save_file_dir = f'{ROOT_DIR}/embedding_model/user_movie_embedding_final.h5'
        assert os.path.exists(embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."
        
        # Loading embedding network weights
        self.embedding_network.load_weights(embedding_save_file_dir)

        # Initialize state representation
        self.srm_ave = StateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1,state_size, 100))])
        self.batch_loss = 0.0

        #define the update rate at which we update the target network
        self.update_interval = 1000
        self.update_counter = 0
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)

        
    # defining epsilon greedy function so our agent can tackle exploration vs exploitation issue
    def epsilon_greedy(self, state):
        #whenever a random value < epsilon we take random action
        if random.uniform(0,1) <= self.epsilon:
            return np.random.randint(self.action_size)

        #then we calculate the Q value 
        Q_values = self.q_network(np.array([state]))##neural n/w is used to predict the action with highest q-value for current state
        return np.argmax(Q_values)#returns the  action with the highest Q-value as predicted by the neural network.
    

    def build_network(self):
        self.q_network(np.zeros((1, 3*self.embedding_dim)))
        self.target_network(np.zeros((1, 3*self.embedding_dim)))
    
    def update_target_network(self):
        q_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(q_weights)):
            target_weights[i] = q_weights[i] * self.gamma + target_weights[i] * (1 - self.gamma)
            self.target_network.set_weights(target_weights)
  
    def update_q_network(self):
        # Check if we have enough samples in the replay buffer to train on
        if self.replay_buffer.get_size() < self.batch_size:
            return 0.0

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(self.batch_size)
        with tf.GradientTape() as g:
            q_values = self.q_network(batch_states)
            next_q_values = self.q_network(batch_next_states)
            next_target_values = self.target_network(batch_next_states)
            target_q_values = np.copy(q_values)
            for i in range(self.batch_size):
                target = batch_rewards[i]
                if not batch_dones[i]:
                    next_action = np.argmax(next_q_values[i])
                    target = target + self.gamma * next_target_values[i][next_action]
                target_q_values[i][batch_actions[i]] = target
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            grads = g.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
        # Periodically update the weights of the target network with the weights of the main network
        if self.update_counter % self.update_interval == 0:
            self.update_target_network()
        # Increment the update counter
        self.update_counter += 1
        return loss
        
    def train(self, env, max_episode_num, load_model=False):
        if load_model:
            self.load_model(os.path.join(self.save_model_weight_dir,"q_network_ddqn.h5"), os.path.join(self.save_model_weight_dir, "target_network_ddqn.h5"))
            print('Completely loaded weights of the networks!')
        episode_precision_history = []
        episode_rewards = []
        episode_loss = []
        for episode in range(max_episode_num):
            # Reward per episode
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            user_id, items_ids, done = env.reset()
            # Reset Environment
            while not done:
                # Observe current state and find action
                # Current state in our case is embedding
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))

                # Output state to SRM
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                #select an action based on the current state using the epsilon-greedy strategy
                action = self.epsilon_greedy(state)

                # Calculate reward & observe new state (in env)
                next_items_ids, reward, done, recommended_items = env.step(action)

                # get next_state
                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                # Store in transition buffer
                self.store_transition(state, action, reward, next_state, done)

                self.batch_loss = self.update_q_network()

                q_loss += self.batch_loss
                items_ids = next_items_ids
                episode_reward += reward
                steps += 1

                if reward > 0:
                    correct_count += 1
                # print(f'recommended items : {len(env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                if done:
                    precision = int(correct_count/steps * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                    episode_precision_history.append(precision)
                    episode_rewards.append(episode_reward)
                    episode_loss.append(q_loss)
    
            if (episode+1)%50 == 0:
                plt.plot(episode_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images/training_precision_%_top_10.png'))

            if (episode+1)%100 == 0 or episode == max_episode_num-1:
                self.save_model(os.path.join(self.save_model_weight_dir, f'q_network_{episode+1}_fixed.h5'),
                          os.path.join(self.save_model_weight_dir, f'target_network_{episode+1}_fixed.h5'))
            # Store the training metrics in pickle file
            with open(os.path.join(self.save_model_weight_dir,'ddqn_agent_train_op.pickle'), 'wb') as f:
                data = (episode_precision_history, episode_rewards, episode_loss)
                pickle.dump(data, f) 


    def save_model(self, q_network_path, target_network_path):
        self.q_network.save_weights(q_network_path)
        self.target_network.save_weights(target_network_path)
        
    def load_model(self, q_network_path, target_network_path):
        self.q_network.load_weights(q_network_path)
        self.target_network.load_weights(target_network_path)
    
    def recommend_item(self, user_id, item_ids):
        # Find embeddings of user and items
        user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(item_ids))
        
        # Represent State
        state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        
        # Get most profitable action
        action = self.q_network(state)

        recommended_item = np.argmax(action)
        return recommended_item
    

class DqnAgent(object):
    def __init__(self, env, users_num, items_num, state_size, action_size, epsilon=0.8, epsilon_min=0.01, epsilon_decay=0.999, embedding_dim=100, hidden_dim=256, learning_rate=0.001, gamma = 0.8, buffer_size=5000,batch_size=64):
        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.state_size = state_size
        self.action_size = action_size

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.replay_buffer = PriorityExperienceReplay(self.buffer_size, self.embedding_dim)

        #epsilon of 0.8 denotes we get 20% random decision
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q network and target network initialization
        self.q_network = DNN(self.embedding_dim, self.hidden_dim, self.action_size)
        self.target_network = DNN(self.embedding_dim, self.hidden_dim, self.action_size)
        
        # Optimizer initialization
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        # Embedding layer
        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)),np.zeros((1,))])
        
        # Save model in the path below
        self.save_model_weight_dir = f"{ROOT_DIR}/rl_model/dqn/trail-{datetime.now().strftime('%Y-%m-%d-%H')}"
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'images'))
        embedding_save_file_dir = f'{ROOT_DIR}/embedding_model/user_movie_embedding_final.h5'
        assert os.path.exists(embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."
        
        # Loading embedding network weights
        self.embedding_network.load_weights(embedding_save_file_dir)

        # Initialize state representation
        self.srm_ave = StateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1,state_size, 100))])
        self.batch_loss = 0.0

        #define the update rate at which we update the target network
        self.update_interval = 1000
        self.update_counter = 0
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)

        
    # defining epsilon greedy function so our agent can tackle exploration vs exploitation issue
    def epsilon_greedy(self, state):
        #whenever a random value < epsilon we take random action
        if random.uniform(0,1) <= self.epsilon:
            return np.random.randint(self.action_size)

        #then we calculate the Q value 
        Q_values = self.q_network(np.array([state]))##neural n/w is used to predict the action with highest q-value for current state
        return np.argmax(Q_values)#returns the  action with the highest Q-value as predicted by the neural network.
    

    def build_network(self):
        self.q_network(np.zeros((1, 3*self.embedding_dim)))
        self.target_network(np.zeros((1, 3*self.embedding_dim)))
    
    def update_target_network(self):
        q_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(q_weights)):
            target_weights[i] = q_weights[i] * self.gamma + target_weights[i] * (1 - self.gamma)
            self.target_network.set_weights(target_weights)
  
    def update_q_network(self):
        # Check if we have enough samples in the replay buffer to train on
        if self.replay_buffer.get_size() < self.batch_size:
            return 0.0

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(self.batch_size)
        with tf.GradientTape() as g:
            q_values = self.q_network(batch_states)
            next_q_values = self.target_network(batch_next_states)
            target_q_values = np.copy(q_values)
            for i in range(self.batch_size):
                target = batch_rewards[i]
                if not batch_dones[i]:
                    target += self.gamma * np.max(next_q_values[i])
                target_q_values[i][batch_actions[i]] = target
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            grads = g.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
     
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
       
        # Periodically update the weights of the target network with the weights of the main network
        if self.update_counter % self.update_interval == 0:
            self.update_target_network()
        # Increment the update counter
        self.update_counter += 1
        return loss

    def train(self, env, max_episode_num, load_model=False):
        if load_model:
            self.load_model(os.path.join(self.save_model_weight_dir,"q_network_dqn.h5"), os.path.join(self.save_model_weight_dir, "target_network_dqn.h5"))
            print('Completely loaded weights of the networks!')
        episode_precision_history = []
        episode_rewards = []
        episode_loss = []
        for episode in range(max_episode_num):
            # Reward per episode
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            user_id, items_ids, done = env.reset()
            # Reset Environment
            while not done:
                # Observe current state and find action
                # Current state in our case is embedding
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))

                # Output state to SRM
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                #select an action based on the current state using the epsilon-greedy strategy
                action = self.epsilon_greedy(state)

                # Calculate reward & observe new state (in env)
                next_items_ids, reward, done, recommended_items = env.step(action)

                # get next_state
                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                # Store in transition buffer
                self.store_transition(state, action, reward, next_state, done)

                self.batch_loss = self.update_q_network()

                q_loss += self.batch_loss
                items_ids = next_items_ids
                episode_reward += reward
                steps += 1

                if reward > 0:
                    correct_count += 1
                # print(f'recommended items : {len(env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                if done:
                    precision = int(correct_count/steps * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                    episode_precision_history.append(precision)
                    episode_rewards.append(episode_reward)
                    episode_loss.append(q_loss)

            if (episode+1)%50 == 0:
                plt.plot(episode_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images/training_precision_%_top_10.png'))

            if (episode+1)%100 == 0 or episode == max_episode_num-1:
                self.save_model(os.path.join(self.save_model_weight_dir, f'q_network_{episode+1}_fixed.h5'),
                              os.path.join(self.save_model_weight_dir, f'target_network_{episode+1}_fixed.h5'))
        # Store the training metrics in pickle file
        with open(os.path.join(self.save_model_weight_dir,'dqn_agent_train_op.pickle'), 'wb') as f:
            data = (episode_precision_history, episode_rewards, episode_loss)
            pickle.dump(data, f) 


    def save_model(self, q_network_path, target_network_path):
        self.q_network.save_weights(q_network_path)
        self.target_network.save_weights(target_network_path)
        
    def load_model(self, q_network_path, target_network_path):
        self.q_network.load_weights(q_network_path)
        self.target_network.load_weights(target_network_path)
    
    def recommend_item(self, user_id, item_ids):
        # Find embeddings of user and items
        user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(item_ids))
        
        # Represent State
        state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        
        # Get most profitable action
        action = self.q_network(state)

        recommended_item = np.argmax(action)
        return recommended_item
