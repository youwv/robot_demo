import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            # state, action, reward, next_state, done

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)

        return torch.tensor(state_list, dtype=torch.float32).to(device), \
            torch.tensor(action_list, dtype=torch.float32).to(device), \
            torch.tensor(reward_list, dtype=torch.float32).unsqueeze(-1).to(device), \
            torch.tensor(next_state_list, dtype=torch.float32).to(device), \
            torch.tensor(done_list, dtype=torch.float32).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.ACTION_HIGH = np.divide([2.9, 2.2, 1.2], 20)
        self.ACTION_LOW = np.divide([-2.9, -1.7, -3.4], 20)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        action = np.clip(action, self.ACTION_LOW, self.ACTION_HIGH)

        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class SAC:
    def __init__(self, abb_env, state_dim, action_dim, gamma, tau, buffer_maxlen, q_lr, policy_lr):

        self.env = abb_env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q1_target_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_target_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)

        self.hard_update(self.q1_target_net, self.q1_net)
        self.hard_update(self.q2_target_net, self.q2_net)
        # Initialize the optimizer
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = torch.tensor(0, dtype=torch.float32)

        # Initialize the buffer
        self.buffer = ReplayBuffer(buffer_maxlen)

    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @staticmethod
    def hard_update(target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)

    def get_action(self, state):
        action = self.policy_net.action(state)

        return action

    def update(self, batch_size, updates, updates_rate):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        with torch.no_grad():
            new_next_action, log_prob = self.policy_net.evaluate(next_state)

            # Soft q  loss
            q1_value_next = self.q1_target_net(next_state, new_next_action)
            q2_value_next = self.q2_target_net(next_state, new_next_action)
            target_min_q_value = torch.min(q1_value_next, q2_value_next)
            target_q_value = reward + (1-done) * self.gamma * (target_min_q_value-self.alpha*log_prob)
            # print reward[0],self.alpha*log_prob[0]
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        q1_value_loss = F.mse_loss(q1_value, target_q_value)
        q2_value_loss = F.mse_loss(q2_value, target_q_value)

        # Policy loss
        new_action, new_log_prob = self.policy_net.evaluate(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        new_q_value_min = torch.min(new_q1_value, new_q2_value)
        policy_loss = (self.alpha*new_log_prob - new_q_value_min).mean()

        # alpha loss
        alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward(retain_graph=True)
        q2_value_loss.backward(retain_graph=True)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        # Update target_Q networks
        if updates % updates_rate == 0:
            self.soft_update(self.q1_net, self.q1_target_net)
            self.soft_update(self.q2_net, self.q2_target_net)


if __name__ == '__main__':
    import virtual_env
    import matplotlib.pyplot as plt
    import moveit_commander

    env = virtual_env.abb_env()
    BUFFER_SIZE = int(30000)
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.01
    EPISODES = 1000
    EP_MAX_STEPS = 100
    STATE_DIM = 6
    ACTION_DIM = 3
    Q_LR = 0.0003
    P_LR = 0.0003

    agent = SAC(env, STATE_DIM, ACTION_DIM, GAMMA, TAU, BUFFER_SIZE, Q_LR, P_LR)
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
        reset_a = [0.0] * 3
        s = env.reset(reset_a)
        ep_reward = 0
        dist_temp = 0.0
        for step in range(EP_MAX_STEPS):
            Count += 1
            a_temp = agent.get_action(s)
            reset_a += a_temp
            next_s, r, d, dist = env.step(reset_a)
            if dist < dist_temp:
                r += 0.3
            else:
                r -= 0.4
            dist_temp = dist
            agent.buffer.push((s, a_temp, r, next_s, d))
            if agent.buffer.buffer_len() > BATCH_SIZE:
                agent.update(BATCH_SIZE, Count, 5)
            s = next_s
            ep_reward += r
            if d:
                break
        ep_reward_list.append(ep_reward)
        if i % 5 == 0:
            print 'Episode:', i, ' Reward:', int(ep_reward), ' dist:', round(dist, 3), 'step:', Count, \
                'alpha:', agent.alpha.cpu().detach().item()

        # test
        if i >= 200 and i % 40 == 0:
            print "-" * 20
            for _ in range(test_frames):
                a_test = [0.0] * 3
                s_test = env.reset(a_test)
                ep_test_rewards = 0
                for k in range(test_max_steps):
                    a_test += agent.get_action(s_test)
                    s_test, r_test, d_test, _ = env.step(a_test)
                    ep_test_rewards += r_test
                    if d_test:
                        break
                test_ep_reward_list.append(ep_test_rewards)
                print 'Episode:', i, ' Reward: %i' % int(ep_reward), 'Test Reward: %i' % int(ep_test_rewards)
            print "-" * 20

    torch.save(agent.policy_net.state_dict(), './folder_for_nn.pth')
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
