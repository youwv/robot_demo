import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import moveit_commander
import ENV
import virtual_env
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
LOG_STD_MIN = -5
LOG_STD_MAX = 2
ACTION_HIGH = np.divide([2.9, 2.2, 1.2], 20)
ACTION_LOW = np.divide([-2.9, -1.7, -3.4], 20)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-2, log_std_max=1, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.ACTION_HIGH = np.divide([2.9, 2.2, 1.2], 40)
        self.ACTION_LOW = np.divide([-2.9, -1.7, -3.4], 40)
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
        return action, #log_prob


model = PolicyNet(6, 3).to(device)
model_dict = model.load_state_dict(torch.load('./folder_for_nn.pth'))
env1 = virtual_env.abb_env()
env2 = ENV.Env()

r = []
for num in range(50):
    env1.get_random_pos()
# print env1.random_pos
env2.random_pos = env1.random_pos
for j in range(50):
    env2.set_target_position()
    action = [0.0] * 3
    s = env2.reset(action)
    ep_r = 0.0
    for i in range(30):
        s = np.reshape(s, (6,))
        # s = torch.tensor(s, dtype=torch.float32).cuda()
        a_temp = model.action(s)# .cpu().detach().numpy()
        action += a_temp
        next_state, reward, done, dist = env2.step(action)
        ep_r += reward
        if done:
            break
        s = next_state
    print ep_r, i, dist
    r.append(ep_r)


plt.plot(r)
plt.show()
moveit_commander.roscpp_shutdown()
