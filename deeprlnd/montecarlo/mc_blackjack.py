
import gym
import numpy as np
import matplotlib.pyplot as plt
import plotUtils
from collections import defaultdict
from tqdm import tqdm

bj_env = gym.make( 'Blackjack-v0' )

## Exploring the environment structure
# an observation is a tuple ( playerSum, dealerSum, hasUsableCard)
# an action is a value of 0 (STICK) or 1 (HIT)
print( 'Observation space: ', bj_env.observation_space )
print( 'Action space : ', bj_env.action_space )
print( '\n\n' )

## Exploring how the environment works and sample episodes
# run 10 episodes and check how the environment works ...
# by using a random policy
for i in range( 3 ) :
    _state = bj_env.reset()
    _return = 0.0
    while True :
        print( 'State: ', _state )
        _action = bj_env.action_space.sample()
        print( 'Action: ', 'HIT' if _action == 1 else 'STICK' )
        _state, _reward, _done, _info = bj_env.step( _action )
        print( 'Reward: ', _reward )
        _return += _reward

        if _done :
            print( 'End Game!!! Return: ', _return )
            if _return > 0 : 
                print( 'You Won :D\n' )
            else : 
                print( 'You lost :/\n' )
            break

## Implementing Monte Carlo prediction
# First, let's use a semy stochoastic policy
# If the player sum is <= 18, we HIT 80% of the time
# If the player sum is > 18, we STICK 20% of the time
def generateEpisodeFromLimitSchochastic( env ) :
    _episode = []
    _state = env.reset()
    while True :
        # policy described earlier
        _probs = [0.8, 0.2] if _state[0] > 18 else [0.2, 0.8]
        _action = np.random.choice( np.arange( 2 ), p = _probs )
        # apply action
        _nextState, _reward, _done, _info = env.step( _action )
        # save into current episode
        _episode.append( ( _state, _action, _reward ) )
        # change to the new state
        _state = _nextState

        if _done :
            break

    return _episode

# Test the current policy
for i in range( 3 ) :
    print( generateEpisodeFromLimitSchochastic( bj_env ) )

# Our implementation of Monte Carlo - Every visit for Q-value function prediction
def mc_predictionQ( env, numEpisodes, fcnEpisodeGenerator, gamma = 1.0 ) :
    # Initialize the dictionaries to store ...
    _G = defaultdict( lambda: np.zeros( env.action_space.n ) )
    _N = defaultdict( lambda: np.zeros( env.action_space.n ) )
    _Q = defaultdict( lambda: np.zeros( env.action_space.n ) )

    # loop over episodes
    for _ in tqdm( range( 1, numEpisodes + 1 ) ) :
        # generate an episode using the policy embedded in the generator fcn
        _episode = fcnEpisodeGenerator( env )
        # reset return sum, and update it backwards
        _Gt = 0
        # Every visit MC prediction
        for t in np.arange( len( _episode ) - 1, -1, -1 ) :
            # extract episode data
            _st  = _episode[t][0]
            _at  = _episode[t][1]
            _rt1 = _episode[t][2]
            # compute return. Because we are traversing backwards we ...
            # are computing the actual return from (st,at) at time t
            _Gt = gamma * _Gt + _rt1
            # update tables
            _N[_st][_at] += 1.0
            _G[_st][_at] += _Gt
            _Q[_st][_at] = _G[_st][_at] / _N[_st][_at]
    
    return _Q

# Instructor's implementation
def mc_predictionQ_instructor( env, numEpisodes, fcnEpisodeGenerator, gamma = 1.0 ) :
    # Initialize the dictionaries to store ...
    _G = defaultdict( lambda: np.zeros( env.action_space.n ) )
    _N = defaultdict( lambda: np.zeros( env.action_space.n ) )
    _Q = defaultdict( lambda: np.zeros( env.action_space.n ) )

    # loop over episodes
    for _ in tqdm( range( 1, numEpisodes + 1 ) ) :
        # generate an episode using the policy embedded in the generator fcn
        _episode = fcnEpisodeGenerator( env )
        # extract states, actions and rewards into separate buffers
        _states, _actions, _rewards = zip( *_episode )
        # calculate discount factors to be applied element-wise
        _discounts = np.array( [ gamma ** i for i in range( len( _rewards ) + 1 ) ] )
        for i, _state in enumerate( _states ) :
            _G[_state][_actions[i]] += sum( _rewards[i:] * _discounts[:-(1 + i)] )
            _N[_state][_actions[i]] += 1.0
            _Q[_state][_actions[i]] = _G[_state][_actions[i]] / _N[_state][_actions[i]]

    return _Q

# obtain the action-value function
Q = mc_predictionQ( bj_env, 500000, generateEpisodeFromLimitSchochastic )

# obtain the corresponding state-value function
V_to_plot = dict( ( k, ( k[0] > 18 ) * ( np.dot( [ 0.8, 0.2 ], v ) ) + 
                       ( k[0] <= 18 ) * ( np.dot( [ 0.2, 0.8 ], v) ) ) \
                  for k, v in Q.items() )

# compare it with the instructor's solution
iQ = mc_predictionQ_instructor( bj_env, 500000, generateEpisodeFromLimitSchochastic )

# obtain the corresponding state-value function
iV_to_plot = dict( ( k, ( k[0] > 18 ) * ( np.dot( [ 0.8, 0.2 ], v ) ) + 
                        ( k[0] <= 18 ) * ( np.dot( [ 0.2, 0.8 ], v) ) ) \
                   for k, v in iQ.items() )

# plot the state-value function
plotUtils.plotBlackjackValues( V_to_plot, False )
plotUtils.plotBlackjackValues( iV_to_plot, False )
plt.show()