import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

env = gym.make('MountainCarContinuous-v0', r

# 超参数
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
N_ACTIONS = env.action_space.shape[0]
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
            nn.Tanh()
        )

    # 定义前向传播
    def forward(self, x):
        x = self.fc(x)
        return x

# 定义DQN
class DQN(object):
    def __init__(self):
        # 定义神经网络
        self.eval_net, self.target_net = Net(), Net()
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # 定义损失函数
        self.loss_fn = nn.MSELoss()
        # 定义记忆池
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        # 记录学习次数
        self.learn_step_counter = 0
        # 记录记忆池中的记忆个数
        self.memory_counter = 0
        # 记录学习曲线
        self.cost_his = []

    # 定义选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)
        x = torch.tensor(x, dtype=torch.float32)
        # 输入状态，得到所有动作的值
        actions_value = self.eval_net.forward(x)
        # 选择值最大的动作
        action = actions_value.detach().numpy()[0]
        return action

    # 定义存储记忆
    def store_transition(self, s, a, r, s_):
        # 将s, a, r, s_打包成一条记忆
        transition = np.hstack((s, a, r, s_))
        # 记录一条记忆
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        # 记录记忆池中的记忆个数
        self.memory_counter += 1

    # 定义学习
    def learn(self):
        # 判断是否更新target_net参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 记录学习次数
        self.learn_step_counter += 1
        # 从记忆池中随机抽取一批记忆
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 将状态，动作，奖励，下一个状态分开
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS])
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS:N_STATES + N_ACTIONS + 1])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 计算当前状态的值
        b_a = torch.tensor(b_a, dtype=torch.int64)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # 计算下一个状态的值
        q_next = self.target_net(b_s_).detach()
        # 计算目标值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # 计算损失函数
        loss = self.loss_fn(q_eval, q_target)
        # 记录学习曲线
        self.cost_his.append(loss.item())
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 定义保存模型
    def save(self):
        torch.save(self.eval_net.state_dict(), 'MountainCarContinuous-v0.pkl')

    # 定义加载模型
    def load(self):
        self.eval_net.load_state_dict(torch.load('MountainCarContinuous-v0.pkl'))
        self.target_net.load_state_dict(torch.load('MountainCarContinuous-v0.pkl'))

# 定义主函数
def main():
    # 定义DQN
    dqn = DQN()
    # 记录奖励
    reward_his = []
    # 记录步数
    step = 0
    # 训练1000次
    for episode in range(1000):
        # 初始化环境
        s, _ = env.reset()
        # 记录奖励
        ep_r = 0
        # 记录步数
        step += 1
        # 最多500步
        for t in range(500):
            # 刷新环境
            env.render()
            # 选择动作
            a = dqn.choose_action(s)
            # 执行动作
            s_, r, done, info, _ = env.step(a)
            # 记录奖励
            ep_r += r
            # 存储记忆
            dqn.store_transition(s, a, r, s_)
            # 记录步数
            step += 1
            # 记忆池中的记忆个数大于2000时，开始学习
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            # 更新状态
            s = s_
            # 显示信息
            print('Episode:', episode, ' Reward: %i' % int(ep_r))
            # 判断是否结束
            if done:
                # 记录奖励
                reward_his.append(ep_r)
                # 输出信息
                print('Episode:', episode, ' Reward: %i' % int(ep_r))
                break
    # 保存模型
    dqn.save()
    # 绘制学习曲线
    plt.plot(np.arange(len(dqn.cost_his)), dqn.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    # 绘制奖励曲线
    plt.plot(np.arange(len(reward_his)), reward_his)
    plt.ylabel('Reward')
    plt.xlabel('training steps')
    plt.show()

# 运行主函数
if __name__ == '__main__':
    main()
