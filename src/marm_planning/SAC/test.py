import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ReplayBeffer():
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
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


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
    def __init__(self, state_dim, action_dim, log_std_min=-2, log_std_max=1, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.ACTION_HIGH = np.divide([2.9, 2.2, 1.2], 20)
        self.ACTION_LOW = np.divide([-2.9, -1.7, -3.4], 20)
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
        state = torch.FloatTensor(state).to(device)
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
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr):

        self.env = env
        self.state_dim = 6
        self.action_dim = 3

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim).to(device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = torch.tensor(0.2, dtype=torch.float32)
        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen)

    def get_action(self, state):
        action = self.policy_net.action(state)

        return action

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.policy_net.evaluate(state)

        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - self.alpha*log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        policy_loss = (self.alpha*log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


def main(env, agent, batch_size):
    reset_action = [0.0] * 3
    Count = 0
    for i in range(max_frames):
        if i % 100 == 0:
            print '******adding some new target_pos:******'
            for num in range(100):
                env.get_random_pos()
        env.set_target_position()
        state = env.reset(reset_action)
        ep_reward = 0
        action = [0.0] * 3
        for step in range(max_steps):
            Count += 1
            a_temp = agent.get_action(state)
            action += a_temp
            next_state, reward, done, dist = env.step(action)
            agent.buffer.push((state, a_temp, reward, next_state, done))
            if agent.buffer.buffer_len() > batch_size:
                agent.update(batch_size)
            state = next_state
            ep_reward += reward
            if done or step == max_steps - 1:
                ep_reward_list.append(ep_reward)
                print 'Episode:', i, ' Reward:', int(ep_reward), ' dist:', round(dist, 3), 'step:', Count,\
                    'alpha:', agent.alpha.cpu().detach().item()

                # test
                if i >= 200 and i % 20 == 0:
                    print "-" * 20
                    for _ in range(test_frames):
                        s_test = env.reset(reset_action)
                        ep_test_rewards = 0
                        a_test = [0.0] * 3
                        for k in range(test_max_steps):
                            a_test += agent.get_action(s_test)
                            s_test, r_test, d_test, _ = env.step(a_test)
                            ep_test_rewards += r_test
                            if d_test:
                                break
                        test_ep_reward_list.append(ep_test_rewards)
                        print 'Episode:', i, ' Reward: %i' % int(ep_reward), 'Test Reward: %i' % int(ep_test_rewards)
                    print "-" * 20
                break
    torch.save(agent.policy_net.state_dict(), './folder_for_nn.pth')
    plt.plot(ep_reward_list)
    img_name = "SAC" + "_epochs" + str(max_frames)
    plt.title(img_name + "_train")
    plt.savefig(img_name + ".png")
    plt.show()
    plt.close()

    plt.plot(test_ep_reward_list)
    plt.title(img_name + "_test")
    plt.savefig(img_name + '_test' + ".png")
    plt.show()
    moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    import virtual_env
    import matplotlib.pyplot as plt
    import moveit_commander

    env = virtual_env.abb_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-4
    value_lr = 3e-4
    policy_lr = 3e-4
    buffer_maxlen = 20000

    max_frames = 600
    max_steps = 150
    test_frames = 1
    test_max_steps = 100
    ep_reward_list = []
    test_ep_reward_list = []
    batch_size = 256

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    main(env, agent, batch_size)
