import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)

        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)

        self.action_scale = torch.FloatTensor((high - low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((high + low) / 2.).to(device)

    def forward(self, state):
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m = self.mean(x)
        s = self.log_std(x)
        s = torch.clamp(s, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return m, s

    def sample(self, state):
        m, s = self.forward(state)
        std = s.exp()
        normal = Normal(m, std)
        a = normal.rsample()

        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias

        # we got a complicated distribution

        logp = normal.log_prob(a)
        logp -= torch.log(self.action_scale * (1 - tanh.pow(2)) + 1e-6)
        logp = logp.sum(1, keepdim=True)

        return action, logp


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Q1
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2


class Memory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, buffer_size, batch_size,
                 gamma, tau,num_updates, update_rate, alpha):

        # Actor Network and Target Network
        self.actor = Actor(state_size, action_size,hidden_dim, high, low).to(device)
        self.actor.apply(self.init_weights)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        # copy weights
        self.hard_update(self.critic_target, self.critic)

        self.state_size = state_size
        self.action_size = action_size

        self.target_entropy = -float(self.action_size)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        self.memory = Memory(buffer_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.update_rate = update_rate
        self.alpha = torch.tensor(alpha, dtype=torch.float32)

        self.iters = 0

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def learn(self, batch):
        for _ in range(self.num_updates):
            state, action, reward, next_state, mask = batch

            state = torch.tensor(state).to(device).float()
            next_state = torch.tensor(next_state).to(device).float()
            reward = torch.tensor(reward).to(device).float().unsqueeze(1)
            action = torch.tensor(action).to(device).float()
            mask = torch.tensor(mask).to(device).float().unsqueeze(1)

            # compute target action
            with torch.no_grad():
                a, logp = self.actor.sample(next_state)

                # compute targets
                Q_target1, Q_target2 = self.critic_target(next_state, a)
                min_Q = torch.min(Q_target1, Q_target2)
                Q_target = reward + self.gamma*mask*(min_Q - self.alpha*logp)

            # update critic
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = F.mse_loss(critic_1, Q_target)
            critic_loss2 = F.mse_loss(critic_2, Q_target)

            # update actor
            pi, log_pi = self.actor.sample(state)
            Q1_pi, Q2_pi = self.critic(state, pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss = (self.alpha*log_pi - min_Q_pi).mean()

            #gradient steps
            self.critic_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update alpha
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # update critic_targe
            self.soft_update(self.critic_target, self.critic)

    def act(self, state):
        state = torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]

    def step(self, state, action, reward, next_state, mask):
        self.iters += 1
        self.memory.push((state, action, reward, next_state, mask))
        if (len(self.memory) >= self.memory.batch_size) and (self.iters % self.update_rate == 0):
            self.learn(self.memory.sample())

    def save(self):
        torch.save(self.actor.state_dict(), "ant_actor.pkl")
        torch.save(self.critic.state_dict(), "ant_critic.pkl")


if __name__ == '__main__':
    import virtual_env
    import matplotlib.pyplot as plt
    import moveit_commander

    env = virtual_env.abb_env()
    BUFFER_SIZE = int(1e4)
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.01
    EPISODES = 800
    EP_MAX_STEPS = 150
    NUM_UPDATES = 1
    UPDATE_RATE = 1
    ENTROPY_COEFFICIENT = 0.2
    STATE_DIM = 6
    ACTION_DIM = 3
    ACTION_HIGH = np.divide([2.9, 2.2, 1.2], 20)
    ACTION_LOW = np.divide([-2.9, -1.7, -3.4], 20)

    agent = Sac_agent(state_size=STATE_DIM, action_size=ACTION_DIM, hidden_dim=256, high=ACTION_HIGH, low=ACTION_LOW,
                      buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
                      num_updates=NUM_UPDATES, update_rate=UPDATE_RATE, alpha=ENTROPY_COEFFICIENT)
    time_start = time.time()
    reset_action = [0.0] * 3
    Count = 0
    test_frames = 1
    test_max_steps = 100
    ep_reward_list = []
    test_ep_reward_list = []
    for i in range(EPISODES):
        if i % 100 == 0:
            print '******adding some new target_pos:******'
            for num in range(100):
                env.get_random_pos()
        env.set_target_position()
        s = env.reset(reset_action)
        ep_reward = 0
        action = [0.0] * 3
        for step in range(EP_MAX_STEPS):
            Count += 1
            a_temp = agent.act(s)
            action += a_temp
            next_s, r, d, dist = env.step(action)
            agent.step(s, a_temp, r, next_s, d)
            s = next_s
            ep_reward += r
            if d or step == EP_MAX_STEPS - 1:
                ep_reward_list.append(ep_reward)
                if i % 5 == 0:
                    print 'Episode:', i, ' Reward:', int(ep_reward), ' dist:', round(dist, 3), 'step:', Count, \
                        'alpha:', agent.alpha.cpu().detach().item()

                # test
                if i >= 200 and i % 20 == 0:
                    print "-" * 20
                    for _ in range(test_frames):
                        s_test = env.reset(reset_action)
                        ep_test_rewards = 0
                        a_test = [0.0] * 3
                        for k in range(test_max_steps):
                            a_test += agent.act(s_test)
                            s_test, r_test, d_test, _ = env.step(a_test)
                            ep_test_rewards += r_test
                            if d_test:
                                break
                        test_ep_reward_list.append(ep_test_rewards)
                        print 'Episode:', i, ' Reward: %i' % int(ep_reward), 'Test Reward: %i' % int(ep_test_rewards)
                    print "-" * 20
                break

    torch.save(agent.actor.state_dict(), './folder_for_nn.pth')
    plt.plot(ep_reward_list)
    img_name = "SAC" + "_epochs" + str(EPISODES)
    plt.title(img_name + "_train")
    plt.savefig(img_name + ".png")
    plt.show()
    plt.close()

    plt.plot(test_ep_reward_list)
    plt.title(img_name + "_test")
    plt.savefig(img_name + '_test' + ".png")
    plt.show()
    moveit_commander.roscpp_shutdown()

