import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
#########################################################################
#NN code
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable 
from tensorboardX import SummaryWriter
from datetime import datetime 
import glob, os 
import argparse


# Define Neural Net model
def weights_init(m):
    classname = m.__class__.name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)

class model_network(nn.Module):
    def __init__(self):
        super(model_network, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden_layer1 = 64
        self.hidden_layer2 = 64
        # self.hidden_layer3 = 50
        # define layers
        self.fc1 = nn.Linear(self.state_space, self.hidden_layer1, bias=False)
        # self.fc2 = nn.Linear(self.hidden_layer1,self.hidden_layer2, bias=False)
        # self.fc3 = nn.Linear(self.hidden_layer2, self.hidden_layer3, bias=False)
        
        self.output = nn.Linear(self.hidden_layer1, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(self.fc1, nn.ReLU(inplace=False), self.output)
        return model(x)

# Read user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, help="Flag to Train or test the network")
parser.add_argument("--load_model", default=True, help="load pretrained model")
args = parser.parse_args()
train = args.train
load_model = args.load_model
# Environment setup
env = gym.make('MountainCar-v0')
env.seed(42);
torch.manual_seed(1); np.random.seed(1)
path = glob.glob(os.path.expanduser('./logs/'))[0]
SummaryWriter = SummaryWriter('{}{}'.format(path, datetime.now().strftime('%b%d_%H-%M-%S')))

# Print some info about the environment
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space.sample)

# Parameters
if train == False:
    epsilon = 0.0
else:
    epsilon = 0.3

discount_factor = 0.99
learning_rate = 0.5
n_successes = 0
n_150 = 0
n_175 = 0
n_100 = 0
max_position = -0.4

NUM_STEPS = 200
NUM_EPISODES = 50000
LEN_EPISODE = 200
reward_history = []

# initialize model
nn_model = model_network()
if load_model == True:
    nn_model.load_state_dict(torch.load('./models/checkpoint_final.pth'), strict=False)
if train == False:
    nn_model.eval()
else :
    nn_model.train()
    # input('a')
    
# Performance metric
recent_reward=[]

loss_fn =  nn.MSELoss() # nn.SmoothL1Loss()
optimizer = optim.Adam(nn_model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

# Run for NUM_EPISODES
for episode in trange(NUM_EPISODES):
    episode_reward = 0
    episode_loss = 0
    curr_state = env.reset()

    for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
        env.render()

        # get Q value for current state
        Q = nn_model(Variable(torch.from_numpy(curr_state).type(torch.FloatTensor)))

        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
        #action = env.action_space.sample()
        # Choosing epsilon-greedy action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0,3)
        else:
            # take the action with maximum Q value
            _, action = torch.max(Q, -1)
            action = action.item()

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       200 steps are done
        next_state, reward, done, _ = env.step(action)

        # This is where your NN/GP code should go
        Q1 = nn_model(Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
        maxQ1, _ = torch.max(Q1,-1)
        
        # Create target vector
        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)
        Q_target[action] = reward + torch.mul(maxQ1.detach(), discount_factor)


        # Update the policy
        loss = loss_fn(Q, Q_target)
        # print(loss)
        # input('a')
        if train:
            # Train the network/GP
            nn_model.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # Record history
        episode_reward += reward
        if train:
            episode_loss += loss.item()

        # Current state for next step
        curr_state = next_state

        # record max position 
        if next_state[0] > max_position:
            max_position = next_state[0]
            SummaryWriter.add_scalar('data/max_position', max_position, episode)

        if done:
            # decrease epsilon value as the number of successful runs increase
            if next_state[0] >= 0.5:
                epsilon *= 0.99
                SummaryWriter.add_scalar('data/epsilon', epsilon, episode)

                # adjust learning rate as model converges
                if train:
                    # if episode%200 == 0:
                    scheduler.step()
                    SummaryWriter.add_scalar('data/learning_rate',optimizer.param_groups[0]['lr'], episode)

                n_successes += 1
                SummaryWriter.add_scalar('data/cumulative_successes', n_successes, episode)
                SummaryWriter.add_scalar('data/success', 1, episode)
            elif next_state[0] < 0.5:
                SummaryWriter.add_scalar('data/success', 0, episode)
            # Record history
            reward_history.append(episode_reward)
            recent_reward.append(episode_reward)
            if len(recent_reward) >100:
                del recent_reward[0]
            if (np.mean(recent_reward) > -175):
                    n_175 +=1
            if (np.mean(recent_reward) > -150):
                n_150 += 1
            if (np.mean(recent_reward) > -100):
                n_100 += 1
            SummaryWriter.add_scalar('data/running_reward', np.mean(recent_reward), episode)

            SummaryWriter.add_scalar('data/episode_reward', episode_reward, episode)
            SummaryWriter.add_scalar('data/episode_loss', episode_loss, episode)
            # if train:
            SummaryWriter.add_scalar('data/position', next_state[0], episode)
            break

            You may want to plot periodically instead of after every episode
            Otherwise, things will slow down
            fig = plt.figure(1)
            plt.clf()
            plt.xlim([0,NUM_EPISODES])
            plt.plot(reward_history,'ro')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Per Episode')
            plt.pause(0.01)
            fig.canvas.draw()
if train:
    torch.save(nn_model.state_dict(), './models/checkpoint_final.pth')
SummaryWriter.close()
env.close()
print('successful episodes: {:d} - {:.4f}%'.format(n_successes, n_successes*100/NUM_EPISODES))
print('num_episodes with reward > -175: {:d} - {:.4f}%'.format(n_175, n_175*100/NUM_EPISODES))
print('num_episodes with reward > -150: {:d} - {:.4f}%'.format(n_150, n_150*100/NUM_EPISODES))
print('num_episodes with reward > -100: {:d} - {:.4f}%'.format(n_100, n_100*100/NUM_EPISODES))

