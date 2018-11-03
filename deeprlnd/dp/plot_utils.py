
import numpy as np
import matplotlib.pyplot as plt

"""
:param V : 2D Array representing a Value function
:param ssdim : state space dimension
"""
def plot_value_function( V, ssdim = ( 4, 4 ) ) :
    _V_table = np.reshape( V, ssdim )

    _fig = plt.figure( figsize = ( 6, 6 ) )
    _ax  = _fig.add_subplot( 111 )
    _im = _ax.imshow( _V_table, cmap = 'cool' )

    for ( j, i ), label in np.ndenumerate( _V_table ) :
        _ax.text( i, j, 
                  np.round( label, 5 ), 
                  ha = 'center', 
                  va = 'center', 
                  fontsize = 14 )

    plt.tick_params( bottom = False, left = False, 
                     labelbottom = False, labelleft = False )
    plt.title( 'State-Value Function' )
    plt.show()