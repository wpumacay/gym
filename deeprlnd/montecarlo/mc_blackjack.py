
import gym
import numpy as np
import matplotlib.pyplot as plt
import plotUtils
from collections import defaultdict
from tqdm import tqdm

bj_env = gym.make( 'Blackjack-v0' )

## PART 1 - Monte Carlo Prediction ######################################################################

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
    _G = defaultdict( lambda: np.zeros( 2 ) )
    _N = defaultdict( lambda: np.zeros( 2 ) )
    _Q = defaultdict( lambda: np.zeros( 2 ) )

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

def mc_predictionQ_constantAlpha( env, numEpisodes, fcnEpisodeGenerator, alpha, gamma = 1.0 ) :
    _Q = defaultdict( lambda: np.zeros( 2 ) )

    for _ in tqdm( range( 1, numEpisodes + 1 ) ) :
        _episode = fcnEpisodeGenerator( env )
        _Gt = 0
        
        for t in np.arange( len( _episode ) - 1, -1, -1 ) :
            _st  = _episode[t][0]
            _at  = _episode[t][1]
            _rt1 = _episode[t][2]

            _Gt = gamma * _Gt + _rt1

            _Q[_st][_at] = _Q[_st][_at] + alpha * ( _Gt - _Q[_st][_at] )
    
    return _Q

def mc_predictionV( env, numEpisodes, fcnEpisodeGenerator, gamma = 1.0 ) :
    _G = defaultdict( lambda: np.zeros( 1 ) )
    _N = defaultdict( lambda: np.zeros( 1 ) )
    _V = defaultdict( lambda: np.zeros( 1 ) )

    for _ in tqdm( range( 1, numEpisodes + 1 ) ) :
        _episode = fcnEpisodeGenerator( env )
        _Gt = 0.0

        for t in np.arange( len( _episode ) - 1, -1, -1 ) :
            _st  = _episode[t][0]
            _at  = _episode[t][1]
            _rt1 = _episode[t][2]

            _Gt = gamma * _Gt + _rt1

            _N[_st] += 1.0
            _G[_st] += _Gt
            _V[_st] = _G[_st] / _N[_st]

    return _V

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

# obtain the action-value function (MC every visit)
Q = mc_predictionQ( bj_env, 500000, generateEpisodeFromLimitSchochastic )
# obtain the corresponding state-value function
V_to_plot = dict( ( k, ( k[0] > 18 ) * ( np.dot( [ 0.8, 0.2 ], v ) ) + 
                       ( k[0] <= 18 ) * ( np.dot( [ 0.2, 0.8 ], v) ) ) \
                  for k, v in Q.items() )

# obtain the action-value function (MC with constant alpha)
aQ = mc_predictionQ_constantAlpha( bj_env, 500000, generateEpisodeFromLimitSchochastic, 0.0125 )
# obtaint the corresponding state-value function
aV_to_plot = dict( ( k, ( k[0] > 18 ) * ( np.dot( [ 0.8, 0.2 ], v ) ) + 
                       ( k[0] <= 18 ) * ( np.dot( [ 0.2, 0.8 ], v) ) ) \
                   for k, v in aQ.items() )

# compare it with the instructor's solution
iQ = mc_predictionQ_instructor( bj_env, 500000, generateEpisodeFromLimitSchochastic )
# obtain the corresponding state-value function
iV_to_plot = dict( ( k, ( k[0] > 18 ) * ( np.dot( [ 0.8, 0.2 ], v ) ) + 
                        ( k[0] <= 18 ) * ( np.dot( [ 0.2, 0.8 ], v) ) ) \
                   for k, v in iQ.items() )

# obtaing the state-value function
V = dict( mc_predictionV( bj_env, 500000, generateEpisodeFromLimitSchochastic ) )

# plot the state-value function
plotUtils.plotBlackjackValues( V_to_plot, False )
plotUtils.plotBlackjackValues( aV_to_plot, False )
plotUtils.plotBlackjackValues( iV_to_plot, False )
plotUtils.plotBlackjackValues( V, False )
plt.show()

#########################################################################################################

## PART 2 - Monte Carlo Control #########################################################################

EPS_MAX = 1.0
EPS_MIN = 0.01
EPISODES_THRESHOLD = 100000

def getEpsFromSchedule( i_episode ) :
    if i_episode < EPISODES_THRESHOLD :
        return EPS_MAX - ( EPS_MAX - EPS_MIN ) * ( i_episode / EPISODES_THRESHOLD )
    else :
        return EPS_MIN
    
def pickActionEpsGreedy( Q, state, eps ) :
    if ( np.random.random() <= ( 1 - eps ) ) and ( state in Q ):
        # greedy action
        return np.argmax( Q[ state ] )
    else :
        # random action
        return np.random.choice( np.arange( 2 ), p = [ 0.5, 0.5 ] )
    
def generateEpisodeEpsGreedy( env, Q, i_episode ) :
    _episode = []
    _state = env.reset()
    # epsilon parameter configuration (linear decay)
    _eps = getEpsFromSchedule( i_episode )
    
    # run environment with e-greedy policy
    while True :
        _action = pickActionEpsGreedy( Q, _state, _eps )
        _nextState, _reward, _done, _info = env.step( _action )
        _episode.append( ( _state, _action, _reward ) )
        _state = _nextState
        if _done :
            break
    
    return _episode
    
def mc_control( env, num_episodes, alpha, gamma = 1.0 ) :
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict( lambda: np.zeros( nA ) )
    policy = defaultdict( lambda: np.random.choice( np.arange( 2 ), p = [ 0.5, 0.5 ] ) )
    # loop over episodes
    for i_episode in tqdm( range( 1, num_episodes + 1 ) ) :
        # sample episode with the e-greedy policy
        _episode = generateEpisodeEpsGreedy( env, Q, i_episode )
        # return
        _Gt = 0.0
        # traverse the whole episode from end to start
        for i in range( len( _episode ) - 1, -1, -1 ):
            _st  = _episode[i][0]
            _at  = _episode[i][1]
            _rt1 = _episode[i][2]
            
            _Gt = gamma * _Gt + _rt1
            Q[_st][_at] = Q[_st][_at] + alpha * ( _Gt - Q[_st][_at] )
            
    # retrieve policy from action-value function
    for s in Q :
        policy[s] = np.argmax( Q[s] )
            
    return policy, Q

# obtain the estimated optimal policy and action-value function
policy, Q = mc_control( bj_env, 500000, 0.02 )
# obtain the corresponding state-value function
V = dict( ( k,np.max( v ) ) for k, v in Q.items() )
# plot the state-value function
plotUtils.plotBlackjackValues( V, False )
# plot the policy
plotUtils.plotPolicy( policy, False )
# show all
plt.show()




#########################################################################################################