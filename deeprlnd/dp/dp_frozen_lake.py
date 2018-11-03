
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

print( 'PART1: Iterative Policy Evaluation ***************' )

"""
:param state  : Current state to apply bellman backup for
:param V      : Table representing the value function
:param env    : Environment (to get the dynamics from)
:param policy : Agent's policy to evaluate
:param gamma  : Discount factor
"""
def bellman_expectation_backup( state, V, env, policy, gamma ) :
    _v = 0.0
    # first part (exp. over action in policy)
    for a in range( env.nA ) :
        _pi_a = policy[state][a]
        # second part (exp. over next state and reward)
        _transitions = env.P[state][a]
        for ( _ptransition, _nextS, _reward, _done ) in _transitions :
            _v += _pi_a * _ptransition * ( _reward + gamma * V[_nextS] )
    return _v


def policy_evaluation( env, policy, gamma=1, theta=1e-8 ):
    V = np.zeros(env.nS)
    
    while True :
        _error = 0.0
        
        for s in range( env.nS ) :
            # cache old value
            _vOld = V[s]
            # do bellman backup
            V[s] = bellman_expectation_backup( s, V, env, policy, gamma )
            # current value
            _vNew = V[s]
            # compute bellman error for this state
            _error = max( _error, abs( _vNew - _vOld ) )
            
        if _error < theta :
            print( 'Converged!' )
            break
    
    return V

_random_policy = np.ones( [ _envWrapper.env.nS, 
                            _envWrapper.env.nA ] ) / _envWrapper.env.nA

import plot_utils

# Evaluate the function
V = policy_evaluation( _envWrapper.env, _random_policy )
# Show the results
plot_utils.plot_value_function( V )

print( 'END PART 1 ***************************************' )

print( 'PART2: Estimating Q(s,a) from V(s) ***************' )

Q = np.zeros( [ _envWrapper.env.nS,
                _envWrapper.env.nA ] )

def compute_q_from_v( state, action, env, V, gamma ) :
    q = 0.0
    # call the one-step dynamics
    _transitions = env.P[state][action]
    for _prob, _nextS, _reward, _done in _transitions :
        q += _prob * ( _reward + gamma * V[_nextS] )
    
    return q

for s in range( _envWrapper.env.nS ) :
    for a in range( _envWrapper.env.nA ) :
        Q[s][a] = compute_q_from_v( s, a, _envWrapper.env, V, 1.0 )

print( 'Estimated Q' )
print( Q )
print( 'Ground truth' )
print( np.array( [[ 0.0147094  , 0.01393978 , 0.01393978 , 0.01317015],
                  [ 0.00852356 , 0.01163091 , 0.0108613  , 0.01550788],
                  [ 0.02444514 , 0.02095298 , 0.02406033 , 0.01435346],
                  [ 0.01047649 , 0.01047649 , 0.00698432 , 0.01396865],
                  [ 0.02166487 , 0.01701828 , 0.01624865 , 0.01006281],
                  [ 0.         , 0.         , 0.         , 0.        ],
                  [ 0.05433538 , 0.04735105 , 0.05433538 , 0.00698432],
                  [ 0.         , 0.         , 0.         , 0.        ],
                  [ 0.01701828 , 0.04099204 , 0.03480619 , 0.04640826],
                  [ 0.07020885 , 0.11755991 , 0.10595784 , 0.05895312],
                  [ 0.18940421 , 0.17582037 , 0.16001424 , 0.04297382],
                  [ 0.         , 0.         , 0.         , 0.        ],
                  [ 0.         , 0.         , 0.         , 0.        ],
                  [ 0.08799677 , 0.20503718 , 0.23442716 , 0.17582037],
                  [ 0.25238823 , 0.53837051 , 0.52711478 , 0.43929118],
                  [ 0.         , 0.         , 0.         , 0.        ]] ) )

# Compute Q(s,a) directly

def q_bellman_expectation_backup( state, action, Q, env, policy, gamma ) :
    q = 0.0
    # extract one-step dynamics for this state-action
    _transitions = env.P[state][action]
    # expectation over one-step dynamics
    for _ptransition, _nextS, _reward, _done in _transitions :
        _v = 0.0
        for a in range( env.nA ) :
            _paction = policy[_nextS][a]
            _v += _paction * Q[_nextS][a]

        q += _ptransition * ( _reward + gamma * _v )

    return q

def q_policy_evaluation( env, policy, gamma = 1, theta = 1e-8 ) :
    Q = np.zeros( [ env.nS, env.nA ] )

    while True :
        _error = 0.0

        for s in range( env.nS ) :
            for a in range( env.nA ) :
                # cache old value
                _qOld = Q[s][a]
                # do bellman backup
                Q[s][a] = q_bellman_expectation_backup( s, a, Q, env, policy, gamma )
                # current value
                _qNew = Q[s][a]
                # compute bellman error
                _error = max( _error, np.abs( _qNew - _qOld ) )
            
        if _error < theta :
            print( 'Converged' )
            break
    
    return Q

Q_bellman = q_policy_evaluation( _envWrapper.env, _random_policy )
print( 'Q from bellman' )
print( Q_bellman )

print( 'END PART 2 ***************************************' )