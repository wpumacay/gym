
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
    _V = 0.0
    # first part (exp. over action in policy)
    for a in range( env.nA ) :
        _pi_a = policy[state][a]
        # second part (exp. over next state and reward)
        _transitions = env.P[state][a]
        for ( _ptransition, _nextS, _reward, _done ) in _transitions :
            _V += _pi_a * _ptransition * ( _reward + gamma * V[_nextS] )
    return _V


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
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