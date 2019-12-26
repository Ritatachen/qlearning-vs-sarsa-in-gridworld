# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:43:27 2019

@author: Rita
"""
# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# create an grid enviroment for our agent


import numpy as np

#read the file 
data = np.genfromtxt('iceWorld.txt',dtype='str')
width = len(data[0])
height = len(data)


#get initial reward for standard grid
initial_reward = np.zeros((width,height))
start=()
goal = ()
icy = []
openstate=[]
hole=[]

# set rewards for different states
# move to hole cost of -50
# all other movements  cost of -1
# get goal recieve 100

for row in range(width):
    for col in range(height):
        if data[row][col] == "O":
            initial_reward[row][col] = -1
            openstate.append((row,col))
        if data[row][col] == "S":
            initial_reward[row][col] = -1
            start=(row,col)
        if data[row][col] == "G":
            initial_reward[row][col] = 100
            goal=(row,col)
        if data[row][col] == "I":
            initial_reward[row][col] = -1
            icy.append((row,col))
        if data[row][col] == "H":
            initial_reward[row][col] = -50
            hole.append((row,col))

#get initial action for standard grid
initial_action = [[0 for i in range(10)] for j in range(10)]
for row in range(width):
    for col in range(height):
        if row ==0 and col ==0:
            initial_action[row][col] = ('D', 'R')
        elif row ==0 and col ==9:
            initial_action[row][col] = ('D', 'L')
        elif row == 9 and col ==0:
            initial_action[row][col] = ('U', 'R')
        elif row == 9 and col ==9:
            initial_action[row][col] = ('U', 'L') 
            
        elif row ==0:
            initial_action[row][col] = ('L', 'D', 'R')        
        
        elif col == 0:
            initial_action[row][col] =('U', 'D','R')       
        elif 1<=row <=8 and 1<=col <=8:
             initial_action[row][col] = ('U', 'D', 'R','L')
        elif col == 9:
            initial_action[row][col] = ('U', 'D', 'L')    
        elif row ==9:
            initial_action[row][col] = ('L', 'U', 'R') 


class Grid: # Environment
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions


  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())



def standard_grid():

  g = Grid(width, height, start)
  rewards = {(rx, cx): c for rx, r in enumerate(initial_reward)\
                for cx, c in enumerate(r)}
  actions = {(rx, cx): c for rx, r in enumerate(initial_action)\
                for cx, c in enumerate(r)}
  del actions[(7,9)]  ## delete goal state action 
  g.set(rewards, actions)
  return g



def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="")
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  " % a, end="")
    print("")



