import random
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(19, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).cuda()
        state = state.view(-1, 2, 200, 200)
        state_feature = self.conv1(state)
        state_feature = self.conv2(state_feature)
        state_feature = self.conv3(state_feature)
        state_feature = self.conv4(state_feature)
        state_feature = self.conv5(state_feature)
        state_feature = self.conv6(state_feature)
        state_feature = state_feature.view(state_feature.size(0), -1)
        x = torch.cat([state_feature, action], 1)
        x = self.linear1(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.a_bound = torch.tensor(np.divide([2.9, 1.90, 2.2], 10), dtype=torch.float32).cuda()
        self.offset = torch.tensor(np.divide([0.0, 0.2, -1.1], 10), dtype=torch.float32).cuda()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d()
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(16, num_actions)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda()
        state = state.view(-1, 2, 200, 200)
        state_feature = self.conv1(state)
        state_feature = self.conv2(state_feature)
        state_feature = self.conv3(state_feature)
        state_feature = self.conv4(state_feature)
        state_feature = self.conv5(state_feature)
        state_feature = self.conv6(state_feature)
        state_feature = state_feature.view(state_feature.size(0), -1)
        x = torch.tanh(self.linear1(state_feature))
        return x

    def get_action(self, state):
        action = self.forward(state)
        action *= self.a_bound
        action += self.offset
        action = action.detach().cpu().numpy()[0]
        return action


class Agent:
    def __init__(self, action_dim, batch_size, value_lr, policy_lr, buffer_size, noise):
        self.value_net = ValueNetwork(action_dim).to(device)
        self.policy_net = PolicyNetwork(action_dim).to(device)

        with torch.no_grad():
            self.target_value_net = ValueNetwork(action_dim).to(device)
            self.target_policy_net = PolicyNetwork(action_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_lr = value_lr
        self.policy_lr = policy_lr

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.value_criterion = nn.MSELoss()

        self.replay_buffer_size = buffer_size
        # self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.action_noise = noise
        self.batch_size = batch_size

    def select_action(self, state, test=False):
        action = self.policy_net.get_action(state)
        if not test:
            action = np.random.normal(action, self.action_noise)
        return action

    def ddpg_update(self, gamma=0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-2):
        with torch.no_grad():

            state, action, reward, next_state, done = transition_saver.dump(batch_size=self.batch_size)
            for index in range(len(action)):
                data = action[index][1:-1]
                data = re.findall('[^\s]+', data, re.S)
                action[index] = map(float, data)

            action = torch.tensor(action, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
            done = torch.tensor(np.float32(done)).unsqueeze(1).to(device)

            next_action = self.target_policy_net(next_state)
            target_value = self.target_value_net(next_state, next_action.detach())

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()
        expected_value = reward + done * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        self.action_noise *= 0.9999


if __name__ == '__main__':
    import env
    import os
    import save_transition

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    env = env.Env()
    ACTION_DIM = 3
    TRAIN_EPISODE = 700
    TRAIN_STEP = 100
    TEST_EPISODE = 1
    TEST_STEP = 100
    COUNT = 0
    BATCH_SIZE = 32
    VALUE_LR = 3e-3
    POLICY_LR = 3e-4
    BUFFER_SIZE = 10000

    transition_saver = save_transition.Transition_Save(buffer_size=BUFFER_SIZE)
    agent = Agent(ACTION_DIM, BATCH_SIZE, VALUE_LR, POLICY_LR, BUFFER_SIZE, noise=1.0)
    TRAIN_REWARD_LIST = []
    TEST_REWARD_LIST = []
    RESET_ACTION = [0.0] * 3
    for i in range(TRAIN_EPISODE):
        if i % 100 == 0:
            print '******add some new target_pos:******'
            for num in range(10):
                env.get_random_pos()
                time.sleep(1)
        env.set_target_position()
        # s is [left_cv_image, right_cv_image]
        s = env.reset(RESET_ACTION)
        ep_reward = 0
        a = RESET_ACTION[:]
        dist_temp = 0.5
        for j in range(TRAIN_STEP):
            a_temp = agent.select_action(s)
            a += a_temp
            s_, r, terminal, dist = env.step(a)
            COUNT += 1
            if dist < dist_temp:
                r += 0.1
            else:
                r -= 0.1
            transition = (s, a_temp, r, s_, terminal)
            transition_saver.save(transition)
            dist_temp = dist
            s = s_
            ep_reward += r
            if transition_saver.sample_number > BUFFER_SIZE:
                agent.ddpg_update()
            if terminal or j == TRAIN_STEP - 1:
                TRAIN_REWARD_LIST.append(ep_reward)
                print 'Episode:', i, ' Reward:', int(ep_reward), ' dist:', round(dist, 3), 'step:', COUNT, \
                    'noise:', agent.action_noise

                # test
                if i >= 300 and i % 20 == 0:
                    print "-" * 20
                    for _ in range(TEST_EPISODE):
                        s_test = env.reset([0.0] * 3)
                        ep_rewards = 0
                        a_test = [0.0] * 3
                        for k in range(TEST_STEP):
                            a_test += agent.select_action(s_test, test=True)
                            s_test, r_test, d, _ = env.step(a_test)
                            ep_rewards += r_test
                            if d:
                                break
                        TEST_REWARD_LIST.append(ep_rewards)
                        print 'Episode:', i, 'Test Reward: %i' % int(ep_rewards)
                    print "-" * 20
                    if i % 50 == 0:
                        torch.save(agent.policy_net.state_dict(), './ddpg_torch.pth')
                        print "save policy_net successfully!"
                break

    import matplotlib.pyplot as plt
    img_name = "ddpg" + "_epochs" + str(TRAIN_EPISODE)
    plt.plot(TRAIN_REWARD_LIST)
    plt.title(img_name + "_train")
    plt.savefig(img_name + ".png")
    plt.show()
    plt.close()

    plt.plot(TEST_REWARD_LIST)
    plt.title(img_name + "_test")
    plt.savefig(img_name + '_test' + ".png")
    plt.show()
    plt.close()
