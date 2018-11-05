
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotBlackjackValues( V, show ) :

    def getZ( x, y, usable_ace ) :
        if ( x, y, usable_ace ) in V :
            return V[ x, y, usable_ace ]
        else :
            return 0

    def getFigure( usable_ace, ax ) :
        # our current sum
        _x_range = np.arange( 11, 22 )
        # dealer's showing
        _y_range = np.arange( 1, 11 )

        _X, _Y = np.meshgrid( _x_range, _y_range )
        _Z = np.array( [ getZ( x, y, usable_ace ) 
                            for x, y in zip( np.ravel( _X ), np.ravel( _Y ) ) ] )
        _Z = _Z.reshape( _X.shape )
        
        _surfplot = ax.plot_surface( _X, _Y, _Z, 
                                     rstride = 1, cstride = 1,
                                     cmap = plt.cm.coolwarm,
                                     vmin = -1.0, vmax = 1.0 )
        ax.set_xlabel( 'Player\'s Current Sum' )
        ax.set_ylabel( 'Dealer\'s Showing Card' )
        ax.set_zlabel( 'State Value' )
        ax.view_init( ax.elev, -120 )

    _fig = plt.figure( figsize = ( 20, 20 ) )
    _ax = _fig.add_subplot( 211, projection = '3d' )
    _ax.set_title( 'Usable Ace' )
    getFigure( True, _ax )
    _ax = _fig.add_subplot( 212, projection = '3d' )
    _ax.set_title( 'No Usable Ace' )
    getFigure( False, _ax )
    if show :
        plt.show()