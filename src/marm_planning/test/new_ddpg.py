import tensorflow as tf
print(tf.__version__)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

env = gym.make("Pendulum-v0").unwrapped
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]
gamma = 0.9
tau = 0.01
replay_buffer = 10000
batch_size = 128


class DDPG(object):

    #first, let's define the init method
    def __init__(self, state_shape, action_shape, high_action_value,):
        
        #define the replay buffer for storing the transitions
        self.replay_buffer = np.zeros((replay_buffer, state_shape * 2 + action_shape + 1), dtype=np.float32)
    
        #initialize the num_transitionsto 0 which implies that the number of transitions in our
        #replay buffer is zero
        self.num_transitions = 0
            
        #start the TensorFlow session
        self.sess = tf.Session()
        
        #we learned that in DDPG, instead of selecting the action directly, to ensure exploration,
        #we add some noise using the Ornstein-Uhlenbeck process. So, we first initialize the noise
        self.noise = 3.0
        
        #initialize the state shape, action shape, and high action value
        self.state_shape, self.action_shape, self.high_action_value = state_shape, action_shape, high_action_value
        
        #define the placeholder for the state
        self.state = tf.placeholder(tf.float32, [None, state_shape], 'state')
        
        #define the placeholder for the next state
        self.next_state = tf.placeholder(tf.float32, [None, state_shape], 'next_state')
        
        #define the placeholder for reward
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
        
        #with the actor variable scope
        with tf.variable_scope('Actor'):

            #define the main actor network which is parameterized by phi. Actor network takes the state
            #as an input and returns the action to be performed in that state
            self.actor = self.build_actor_network(self.state, scope='main', trainable=True)
            
            #Define the target actor network which is parameterized by phi dash. Target actor network takes
            #the next state as an input and returns the action to be performed in that state
            target_actor = self.build_actor_network(self.next_state, scope='target', trainable=False)
            
        #with the critic variable scope
        with tf.variable_scope('Critic'):
            
            #define the main critic network which is parameterized by theta. Critic network takes the state
            #and also the action produced by the actor in that state as an input and returns the Q value
            critic = self.build_critic_network(self.state, self.actor, scope='main', trainable=True)
            
            #Define the target critic network which is parameterized by theta dash. Target critic network takes
            #the next state and also the action produced by the target actor network in the next state as
            #an input and returns the Q value
            target_critic = self.build_critic_network(self.next_state, target_actor, scope='target', 
                                                      trainable=False)
            
        
        #get the parameter of the main actor network, phi
        self.main_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/main')
        
        #get the parameter of the target actor network, phi dash
        self.target_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
    
        #get the parameter of the main critic network, theta
        self.main_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/main')
        
        #get the parameter of the target critic networ, theta dash
        self.target_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        #perform the soft replacement and update the parameter of the target actor network and
        #the parameter of the target critic network
        self.soft_replacement = [

            [tf.assign(phi_, tau*phi + (1-tau)*phi_), tf.assign(theta_, tau*theta + (1-tau)*theta_)]
            for phi, phi_, theta, theta_ in zip(self.main_actor_params, self.target_actor_params,
                                                self.main_critic_params, self.target_critic_params)
            ]
        
        #compute the target Q value, we learned that the target Q value can be computed as the
        #sum of reward and discounted Q value of next state-action pair
        y = self.reward + gamma * target_critic
        
        #now, let's compute the loss of the critic network. The loss of the critic network is the mean
        #squared error between the target Q value and the predicted Q value
        MSE = tf.losses.mean_squared_error(labels=y, predictions=critic)
        
        #train the critic network by minimizing the mean squared error using Adam optimizer
        self.train_critic = tf.train.AdamOptimizer(0.01).minimize(MSE, name="adam-ink", var_list = self.main_critic_params)
        
        
        #We learned that the objective function of the actor is to generate an action that maximizes
        #the Q value produced by the critic network. We can maximize the above objective by computing gradients 
        #and by performing gradient ascent. However, it is a standard convention to perform minimization rather 
        #than maximization. So, we can convert the above maximization objective into the minimization
        #objective by just adding a negative sign.
        
        
        #now we can minimize the actor network objective by computing gradients and by performing gradient descent
        actor_loss = -tf.reduce_mean(critic)    
           
        #train the actor network by minimizing the loss using Adam optimizer
        self.train_actor = tf.train.AdamOptimizer(0.001).minimize(actor_loss, var_list=self.main_actor_params)
            
        #initialize all the TensorFlow variables:
        self.sess.run(tf.global_variables_initializer())

       
    #let's define a function called select_action for selecting the action with the noise to ensure exploration
    def select_action(self, state):
        
        #run the actor network and get the action
        action = self.sess.run(self.actor, {self.state: state[np.newaxis, :]})[0]
        
        #now, we generate a normal distribution with mean as action and standard deviation as the
        #noise and we randomly select an action from this normal distribution
        action = np.random.normal(action, self.noise)
        
        #we need to make sure that our action should not fall away from the action bound. So, we
        #clip the action so that they lie within the action bound and then we return the action
        action =  np.clip(action, action_bound[0],action_bound[1])
        
        return action
        
    #now, let's define the train function
    def train(self):
        
        #perform the soft replacement
        self.sess.run(self.soft_replacement)
        
        #randomly select indices from the replay buffer with the given batch size
        indices = np.random.choice(replay_buffer, size=batch_size)
        
        #select the batch of transitions from the replay buffer with the selected indices
        batch_transition = self.replay_buffer[indices, :]

        #get the batch of states, actions, rewards, and next states
        batch_states = batch_transition[:, :self.state_shape]
        batch_actions = batch_transition[:, self.state_shape: self.state_shape + self.action_shape]
        batch_rewards = batch_transition[:, -self.state_shape - 1: -self.state_shape]
        batch_next_state = batch_transition[:, -self.state_shape:]

        #train the actor network
        self.sess.run(self.train_actor, {self.state: batch_states})
        
        #train the critic network
        self.sess.run(self.train_critic, {self.state: batch_states, self.actor: batch_actions,
                                          self.reward: batch_rewards, self.next_state: batch_next_state})
        

    #now, let's store the transitions in the replay buffer
    def store_transition(self, state, actor, reward, next_state):

        #first stack the state, action, reward, and next state
        trans = np.hstack((state,actor,[reward],next_state))
        
        #get the index
        index = self.num_transitions % replay_buffer
        
        #store the transition
        self.replay_buffer[index, :] = trans
        
        #update the number of transitions
        self.num_transitions += 1
        
        #if the number of transitions is greater than the replay buffer then train the network
        if self.num_transitions > batch_size:
            self.noise *= 0.99995
            self.train()
            

    def build_actor_network(self, state, scope, trainable):
        
        #we define a function called build_actor_network for building the actor network. The
        #actor network takes the state and returns the action to be performed in that state
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(state, 30, activation = tf.nn.tanh, name = 'layer_1', trainable = trainable)
            actor = tf.layers.dense(layer_1, self.action_shape, activation = tf.nn.tanh, name = 'actor', trainable = trainable)     
            return tf.multiply(actor, self.high_action_value, name = "scaled_a")  


    def build_critic_network(self, state, actor, scope, trainable):
        
        #we define a function called build_critic_network for building the critic network. The
        #critic network takes the state and the action produced by the actor in that state and returns the Q value
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.state_shape, 30], trainable = trainable)
            w1_a = tf.get_variable('w1_a', [self.action_shape, 30], trainable = trainable)
            b1 = tf.get_variable('b1', [1, 30], trainable = trainable)
            net = tf.nn.tanh( tf.matmul(state, w1_s) + tf.matmul(actor, w1_a) + b1 )

            critic = tf.layers.dense(net, 1, trainable = trainable)
            return critic


ddpg = DDPG(state_shape, action_shape, action_bound[1])

num_episodes = 300

num_timesteps = 500

#for each episode
for i in range(num_episodes):
    
    #initialize the state by resetting the environment
    state = env.reset()
    
    #initialize the return
    Return = 0
    
    #for every step
    for j in range(num_timesteps):
        
        #render the environment
        # env.render()

        #select the action
        action = ddpg.select_action(state)
        
        #perform the selected action
        next_state, reward, done, info = env.step(action)
        
        #store the transition in the replay buffer
        ddpg.store_transition(state, action, reward, next_state)
      
        #update the return
        Return += reward
    
        #if the state is the terminal state then break
        if done:
            break
    
        #update the state to the next state
        state = next_state

    #print the return for every 10 episodes
    if i %10 ==0:
         print("Episode:{}, Return: {}".format(i,Return))


