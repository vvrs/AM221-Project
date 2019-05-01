#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import gym
import random
import math
import time
import os.path

import cv2

import matplotlib.pyplot as plt
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind

import json

ATTACK = "bim"

plt.style.use('ggplot')

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)

# directory = './vid_fgsm/'
# directory = './vid_bim/'
directory = './vid_pgd/'
# directory = './PongVideos/'
env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%20==0, force=True)


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.0001
num_episodes = 500
gamma = 0.99

hidden_layer = 512

replay_mem_size = 100000
batch_size = 32

update_target_frequency = 2000

double_dqn = True

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 18

clip_error = True
normalize_image = True

file2save = 'pong_save.pth'
save_model_frequency = 10000
resume_previous_training = True

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

def load_model():
    return torch.load(file2save)

def save_model(model):
    torch.save(model.state_dict(), file2save)
    
def preprocess_frame(frame):
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(1)
    
    return frame

def saveimage(image_,filename):
    image_ = image_.squeeze().cpu().detach().numpy();
    plt.imsave("{}.jpg".format(filename),cv2.resize(image_,dsize=(256,256),interpolation=cv2.INTER_AREA))


def plot_results():
    plt.figure(figsize=(12,5))
    plt.title("Rewards")
    plt.plot(rewards_total, alpha=0.6, color='red')
    plt.savefig("Pong-results.png")
    plt.close()

####################### ATTACKS ##########################################
def random_attack(image, epsilon_, data_grad):
    # Collect the element-wise sign of the data gradient
    # sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon_*torch.empty(image.shape).uniform_(-128, 127).to(device, dtype=torch.float32)
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image
def fgsm_attack(image, epsilon_, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon_*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def bim_attack(model, criterion, image, epsilon, alpha, iters=0):

    img_bim = torch.tensor(image.data, requires_grad=True)    
    
    if(iters==0):
        iters = int(np.ceil(min(epsilon+4, 1.25*epsilon)))
    else:
        if device.type == "cuda":
            delta_init = torch.from_numpy(np.random.uniform(-epsilon, epsilon, image.shape)).type(torch.cuda.FloatTensor)
        else:
            delta_init = torch.from_numpy(np.random.uniform(-epsilon, epsilon, image.shape)).type(torch.FloatTensor)
        img_bim = torch.tensor(img_bim.data+delta_init, requires_grad = True)
        clipped_delta = torch.clamp(img_bim.data-image.data, -epsilon, epsilon)
        img_bim = torch.tensor(image.data+clipped_delta, requires_grad = True)

    
    for i in range(iters):
        action_from_nn = model(img_bim)
        action_from_nn = F.softmax(action_from_nn)
        logits =  action_from_nn
    
        logits_onehot = torch.zeros(logits.shape).to(device, dtype=torch.float32)
        logits_onehot.scatter_(-1,logits.argmax().view(1,1),1)

        loss = criterion(action_from_nn.view(-1),logits_onehot.view(-1))
        loss.backward()
        delta = alpha*torch.sign(img_bim.grad.data)
        img_bim = torch.tensor(img_bim.data + delta, requires_grad=True)
        clipped_delta = torch.clamp(img_bim.data-image.data, -epsilon, epsilon)
        img_bim = torch.tensor(image.data+clipped_delta, requires_grad = True)
    return img_bim

############################################################################

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = ( self.position + 1 ) % self.capacity
        
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
        
        
    def __len__(self):
        return len(self.memory)
        
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.advantage1 = nn.Linear(7*7*64,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)
        
        self.value1 = nn.Linear(7*7*64,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        # self.activation = nn.softmax();
        
        
    def forward(self, x):
        
        if normalize_image:
            x = x / 255
        
        # Takes Image
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
        
        # Flatten Image
        output_conv = output_conv.view(output_conv.size(0), -1)
        
        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        
        output_value = self.value1(output_conv)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)
        
        output_final = output_value + output_advantage - output_advantage.mean()

        # output is a vector with 1x#actions, can be used for FSGM
        return output_final
    
class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.number_of_frames = 0
        
        if resume_previous_training and os.path.exists(file2save):
            print("Loading previously saved model ... ")
            self.nn.load_state_dict(load_model())
        
        self.bce_loss = nn.BCELoss()
        # self.bce_loss = nn.MSELoss()
        
    def select_action(self,state,epsilon, isAttackActive=False, attackEpsilon = 0.):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            # with torch.no_grad():
                
            state = preprocess_frame(state)

            state.requires_grad = True
            
            action_from_nn = self.nn(state)
            # print(action_from_nn)
            action = torch.max(action_from_nn,1)[1]
            # print("[ORIGINAL ACTION]:",action)
            action = action.item()


            if isAttackActive:
                # random_action = env.action_space.s`ample()
                # random_action = 0.5*torch.ones(action_from_nn.shape).to(device, dtype=torch.float32)
                action_from_nn = F.softmax(action_from_nn)
                logits =  action_from_nn
                # print("[Logits argmax]:",logits.argmax())
                logits_onehot = torch.zeros(logits.shape).to(device, dtype=torch.float32)
                # print("[Logits onehot shape]:",logits_onehot.shape)
                logits_onehot.scatter_(-1,logits.argmax().view(1,1),1)
                # logits = (logits > 0.5).float()
                # print("[random_action]",logits_onehot)

                # print("[LOG]:",action_from_nn.shape, random_action.shape)
                loss = self.bce_loss(action_from_nn.view(-1),logits_onehot.view(-1))
                # print("[Loss]:",loss.item())
                loss.backward()

                # After attack
                # perturb_state = fgsm_attack(state, attackEpsilon, state.grad.data)
                # perturb_state = random_attack(state, attackEpsilon, state.grad.data)
                # perturb_state = bim_attack(self.nn,self.bce_loss,state,attackEpsilon, alpha=attackEpsilon/2, iters=0)
                perturb_state = bim_attack(self.nn,self.bce_loss,state,attackEpsilon, alpha=attackEpsilon/2, iters=1)
                # print("[STATE]:",state.max())
                action_from_nn = self.nn(perturb_state)
                # print("[action from nn]",action_from_nn)
                # print("[ACTIONS]:",action_from_nn.argmax())
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            action = env.action_space.sample()
        # vutils.save_image(state,"sample_fgsm.jpg", nrow=1,normalize=True)
        # print("[Perturbed Action]:",action)
        # # vutils.save_image(perturb_state,"sample_perturbed_fgsm.jpg", nrow=1,normalize=True)
        # saveimage(state,"sample_original_pgd");
        # saveimage(perturb_state,"sample_perturbed_pgd");
        # saveimage(perturb_state-state,"sample_noise_pgd");
        # vutils.save_image(perturb_state-state,"sample_noise_fgsm.jpg", nrow=1,normalize=True)
        # quit()
        return action
    
    def optimize(self):
        
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        
        state = [ preprocess_frame(frame) for frame in state ] 
        state = torch.cat(state)
        
        new_state = [ preprocess_frame(frame) for frame in new_state ] 
        new_state = torch.cat(new_state)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        
        
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
  
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.number_of_frames % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        if self.number_of_frames % save_model_frequency == 0:
            save_model(self.nn)
        
        self.number_of_frames += 1
        
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])

        
# def bim_attack(model, criterion, image, label, epsilon, alpha, iters=0):

#     img_bim = torch.tensor(image.data, requires_grad=True)    
    
#     if(iters==0):
#         iters = int(np.ceil(min(epsilon+4, 1.25*epsilon)))
#     else:
#         if device.type == "cuda":
#             delta_init = torch.from_numpy(np.random.uniform(-epsilon, epsilon, image.shape)).type(torch.cuda.FloatTensor)
#         else:
#             delta_init = torch.from_numpy(np.random.uniform(-epsilon, epsilon, image.shape)).type(torch.FloatTensor)
#         img_bim = torch.tensor(img_bim.data+delta_init, requires_grad = True)
#         clipped_delta = torch.clamp(img_bim.data-image.data, -epsilon, epsilon)
#         img_bim = torch.tensor(image.data+clipped_delta, requires_grad = True)

    
#     for i in range(iters):
#         output = model(img_bim)
#         loss = criterion(output, label)
#         loss.backward()
#         delta = alpha*torch.sign(img_bim.grad.data)
#         img_bim = torch.tensor(img_bim.data + delta, requires_grad=True)
#         clipped_delta = torch.clamp(img_bim.data-image.data, -epsilon, epsilon)
#         img_bim = torch.tensor(image.data+clipped_delta, requires_grad = True)
#     return img_bim

qnet_agent = QNet_Agent()

start_time = time.time()

images = []


# ---------------------------------------------------#
data_logger = {}
EPSILONS = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
NUM_EXP = 5

for e in EPSILONS:    
    print("\n\n[EPSILON]:{}\n\n".format(e))
    episode_scores = []
    cfmat = np.zeros([6,6])
    for i_episode in range(NUM_EXP):
        state = env.reset()
        
        score = 0
        #for step in range(100):
        while True:
            
            #action = env.action_space.sample()
            # print(state.shape)
            # print(type(state))
            # action = qnet_agent.select_action(state, 0, None_perturb)
            action = qnet_agent.select_action(state, 0, isAttackActive=True, attackEpsilon=e)
            action_true = qnet_agent.select_action(state, 0, isAttackActive=False, attackEpsilon=e)
            cfmat[action_true][action] += 1
            state, reward, done, info = env.step(action)\
            # time.sleep(0.05)
            env.render()
            score += reward

            
            
            if done:
                print("EPISODE: {}, SCORE: {}".format(i_episode,score))
                episode_scores.append(score)
                break
    plt.imsave("cfmat_pgd/{}.png".format(e),cv2.resize(cfmat,dsize=(256,256),interpolation=cv2.INTER_AREA))
    data_logger.update({e:episode_scores})
env.close()
env.env.close()
print(data_logger)
json.dump(data_logger,open("results_pgd.json",'w'))