
import gym
import numpy as np
import matplotlib.pyplot as plt

_envWrapper = gym.make( 'FrozenLake-v0' )

print( 'ObsSpace: ', _envWrapper.env.observation_space )
print( 'ActSpace: ', _envWrapper.env.action_space )

print( 'Number of states: ', _envWrapper.env.nS )
print( 'Number of actions: ', _envWrapper.env.nA )

print( 'checking one-step dynamics' )
print( 'P(s\', r | s = 1, a = 0)' )
print( 'prob - next_state - reward - done' )
print( _envWrapper.env.P[1][0] )