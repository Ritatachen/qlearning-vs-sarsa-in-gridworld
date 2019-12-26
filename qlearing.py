# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:04:33 2019

@author: Rita
"""


import numpy as np
import matplotlib.pyplot as plt
from enviroment import standard_grid,print_values, print_policy


data = np.genfromtxt('iceWorld.txt',dtype='str')
width = len(data[0])
height = len(data)

start=()
goal = ()
icy = []
openstate=[]
hole=[]
for row in range(width):
    for col in range(height):
        if data[row][col] == "O":
            openstate.append((row,col))
        if data[row][col] == "S":
            start=(row,col)
        if data[row][col] == "G":
            goal=(row,col)
        if data[row][col] == "I":
            icy.append((row,col))
        if data[row][col] == "H":
            hole.append((row,col))


GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
alpha = 1
episode = 2000
eps=0.9

grid = standard_grid()
print("rewards:")
print_values(grid.rewards, grid)



def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def eps_greedy_action(Q, s, eps):
#  # decay the epsilon value until it reaches the threshold of 0.009

    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a random action
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return max_dict(Q[s])[0]



def icy_move(action):
    if action == 'U':
      return np.random.choice(['L', 'R', 'U'],p =[0.1, 0.1, 0.8])
    elif action == 'D':
      return np.random.choice(['L', 'R', 'D'],p =[0.1, 0.1, 0.8])
    elif action == 'R':
     return np.random.choice(['U', 'D', 'R'],p =[0.1, 0.1, 0.8])
    elif action == 'L':
     return np.random.choice(['U', 'D', 'L'],p =[0.1, 0.1, 0.8])    

def undo_move(action):
    if action == 'U':
        return 'D'
    elif action == 'D':
        return 'U'
    elif action == 'R':
        return 'L'
    elif action == 'L':
        return'R'


def action_onstate(Q,s2,eps,a):
    if s2 in icy:
        a2 =  eps_greedy_action(Q, s2, eps)
        return icy_move(a2)
    elif s2 in openstate:
        return eps_greedy_action(Q, s2, eps)
    elif s2 == start:
        return eps_greedy_action(Q, s2, eps)
    elif s2 in hole:
        return undo_move(a)


def policy_onaction(s,a):
   if s in icy or s in openstate:
       return a
   elif s == start:
       return "S"
   elif s in hole:
       return "H"



#
#def run_episode(gird, Q, num_episodes=100):
#    totalreward = []
#    s = start # start state
#    grid.set_state(s)
#    for _ in range(num_episodes):
#        game_reward = 0
#
#        while not grid.game_over():
#            # select a greedy action
#            a = max_dict(Q[s])[0] 
#            r = grid.move(a) 
#            s2 = grid.current_state()                         
#            s = s2
#            game_reward += r
#            if grid.game_over():
#                totalreward.append(game_reward)
#                
##                policy = {}
##                for s in grid.actions.keys():
##                    if s in icy or openstate:
##                        a = max_dict(Q[s])[0]
##                        policy[s] = a
##                    elif s in hole:
##                        a = "H"
##                        policy[s] = a
##                    elif s in start:
##                        a = "S"               
##                        policy[s] = a
##                print("policy %d: " % it)
##                print_policy(policy, grid)
#                s = start # start state
#                grid.set_state(s)
#    print('Mean score: %.3f of %i games!'%(np.mean(totalreward), num_episodes))
#    return np.mean(totalreward)



Q = {}
states = grid.all_states()
for s in states:
  Q[s] = {}
  for a in ALL_POSSIBLE_ACTIONS:
    Q[s][a] = 0
      
# initial Q values for all states in grid
print(Q)


policy = {}
for s in grid.actions.keys():
  policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

print("initial policy:")
print_policy(policy, grid)


    
# repeat until convergence
t = 1.0
totalreward = []
policy_reward = []
for it in range(1,episode+1):        
    if it % 10 == 0:
        t += 1        
      # instead of 'generating' an epsiode, we will PLAY
      # an episode within this loop
    if it % 100 == 0:
        print("iteration:", it)
        print("initial policy:",it)
        print_policy(policy, grid)

    s = start # start state
#    print(s)
    grid.set_state(s)
#    print(grid.set_state(s))
  
    reward = 0
    
    
    
    if eps > 0.009:
        eps =0.9/t
#        print(eps)
  

#    policy[s] = a
    while not grid.game_over(): 
        a =  eps_greedy_action(Q, s, eps)
#    print("a",a)
        policy[s] = policy_onaction(s,a)

        r = grid.move(a) 
#        print("rewards of a1",r)
        s2 = grid.current_state()
#        print("s2",s2)        
        a2 = action_onstate(Q,s2,eps,a)
#        a2 = np.argmax(Q[s2])

#        policy[s2] = policy_onaction(s2,a2)
#        print("a2",a2)       
        # we will update Q(s,a) AS we experience the episode
#        alpha = ALPHA / update_counts_sa[s][a]
#        update_counts_sa[s][a] += 0.1
        #print("update_counts_sa[s][a]",update_counts_sa[s][a])
#        old_qsa = Q[s][a]
        if grid.game_over():
            Q[s][a] = Q[s][a] + alpha*(r  - Q[s][a])

        else:
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*np.max(Q[s2][a2]) - Q[s][a]) 
#        print("Q[s][a]",Q[s][a])        
        # next state becomes current state        
        
        s = s2
        a = a2
        reward+=r
    policy_reward.append(reward)
    
    
#np.savetxt('qlearning_reward.txt', policy_reward)       



plt.plot(policy_reward,color="b")
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.legend(['qlearning: rewards vs episodes'])  