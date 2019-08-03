

import numpy    as np
import seaborn  as sb 

from matplotlib import pyplot as plt 
import matplotlib.ticker as mtick

def customaxis( ax, 
                position    = 'left',
                color       = 'black',
                label       = None,
                location    = None, 
                scale       = 'linear',
                limits      = None,
                lw          = 1,
                size        = 12,
                pad         = 1.0,
                full_nrs    = True, 
                ):
    """
    Whenever we are plotting with different y axis, it is tedious to set the color of the different axes, the ticks etc. 
    This function takes care of all these problems at once. 

    input:
    -----
    ax:         axis instance
    position:   either 'left', 'right', 'top' or 'bottom'
    color:      what color the axis should have
    label:      the x or y label (None means no label)
    location:   the position of the axis wrt the plotting figure (None means default)
    scale:      the scale of the axis ('linear' or 'log')
    limits:     if not None, then a tuple of minimum and maximum for the axis limits
    lw:         linewidht of axes
    size:       size of labels
    pad:        distance of labels and ticks to the axes
    full_nrs:   If True, show aboslute numbers on the y axis
    """

    assert(position in ['left','right','top','bottom']),"invalid position"

    if position=='left': 
        ax.spines['left'].set_linewidth(lw)
        ax.spines['left'].set_color(color)
        if location is None: location = 0 
        ax.spines['left'].set_position(('axes',location))
        ax.yaxis.tick_left()
        ax.tick_params(axis='y',color=color)
        [i.set_color(color) for i in ax.get_yticklabels()]   
        if label is not None:
            ax.set_ylabel(label,color=color,fontsize=size,)
            ax.yaxis.set_label_position("left")
        ax.set_yscale(scale) 
        if limits is not None: ax.set_ylim(limits)  

    elif position=='right':  
        ax.spines['right'].set_linewidth(lw)
        ax.spines['right'].set_color(color)
        if location is None: location = 1 
        ax.spines['right'].set_position(('axes',location))
        ax.yaxis.tick_right()
        ax.tick_params(axis='y',color=color)
        [i.set_color(color) for i in ax.get_yticklabels()]    
        if label is not None:
            ax.set_ylabel(label,color=color,fontsize=size,)
            ax.yaxis.set_label_position("right")
        ax.set_yscale(scale)       
        if limits is not None: ax.set_ylim(limits)    

    elif position=='bottom':
        ax.spines['bottom'].set_linewidth(lw)
        ax.spines['bottom'].set_color(color)
        if location is None: location = 0 
        ax.spines['bottom'].set_position(('axes',location))
        ax.xaxis.tick_bottom()
        ax.tick_params(axis='x',color=color)
        [i.set_color(color) for i in ax.get_xticklabels()] 
        if label is not None:
            ax.set_xlabel(label,color=color,fontsize=size,)
            ax.xaxis.set_label_position("bottom")
        ax.set_xscale(scale)        
        if limits is not None: ax.set_xlim(limits)       

    else:
        ax.spines['top'].set_linewidth(lw)
        ax.spines['top'].set_color(color)
        if location is None: location = 1 
        ax.spines['top'].set_position(('axes',location))
        ax.xaxis.tick_top()
        ax.tick_params(axis='x',color=color)
        [i.set_color(color) for i in ax.get_xticklabels()]
        if label is not None:
            ax.set_xlabel(label,color=color,fontsize=size,)
            ax.xaxis.set_label_position("top")
        ax.set_xscale(scale)          
        if limits is not None: ax.set_xlim(limits)  

    if full_nrs: ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))





def plot_with_smart_legend(hi) :
    """

    """

    # plt the different X and chi lines: all countries in one plot 
    ####################################################################################################################    
    plt.figure(figsize=(12,7))

    X_axis      = plt.gca()
    chi_axis    = X_axis.twinx()
    colors      = sb.color_palette("bright",2)
    markers     = ['s','o','D','H','^','x','1','p','*','X','2','4']
    fontsize    = 14

    lines  = []
    labels = []

    for i,(whatever) in enumerate( whatevers ):

        ax.plot(whatever, marker=markers[i])
        axt.plot(whatever)

        p       = plt.Line2D( (0,1), (0,0), color='black', linestyle='--',
                              fillstyle='none', marker=markers[i], markersize=12, )
        l       = 'exports from %s to %s'%(c1,c2)
        lines  += [ p ]
        labels += [ l ]

    X_axis.legend( lines, labels, loc='best', fontsize=fontsize, title='country pairs' )

    chi_axis.grid(False)

    # add properly colored axis
    ################################################################################################################
    customaxis(	    ax          = X_axis,
                    position 	= 'left',
                    color		= colors[0],
                    label 		= r'trade matrix $X_{ik}$ in billion USD',
                    scale 		= 'linear',
                    size		= fontsize,
                    full_nrs    = False,
                    )

    customaxis(	    ax          = chi_axis,
                    position 	= 'right',
                    color		= colors[1],
                    label 		= r'ease matrix $\chi_{ik}$',
                    scale 		= 'linear',
                    size		= fontsize,
                    full_nrs    = False,
                    )

    # add title and save to the output
    ################################################################################################################
    plt.savefig( folder + 'all_together.pdf', bbox_inches='tight' )
    plt.close()








def three_y_axes(hi):

    _  = plt.figure(figsize=(12,4))
    ax = plt.gca()
    axt1 = ax.twinx()
    axt2 = ax.twinx()

    ts1 = pd.Series([1,2,3])
    ts2 = pd.Series([10, 11, 12])
    ts3 = pd.Series([100, 110, 120])

    ts1.plot(ax=ax, color='blue')
    ts2.plot(ax=axt1, color='red')
    ts3.plot(ax=axt2, color='green')

    customaxis(     ax          = ax,
                    position    = 'left',
                    color       = 'blue',
                    label       = r'blue line',
                    scale       = 'linear',
                    size        = 12,
                    full_nrs    = False,
                    location    = 0.0, 
                    )

    customaxis(     ax          = axt1,
                    position    = 'right',
                    color       = 'green',
                    label       = r'green line',
                    scale       = 'linear',
                    size        = 12,
                    full_nrs    = False,
                    location    = 1.0, 
                    )

    customaxis(     ax          = axt2,
                    position    = 'right',
                    color       = 'red',
                    label       = r'red line',
                    scale       = 'linear',
                    size        = 12,
                    full_nrs    = False,
                    location    = 1.1, 
                    )    



def circle_plot( a_seller_pos, a_buyer_pos, a_color=None, seller_ind=None, ax=None ):
    """
    Plot a circle of length 2pi and annotate the location of buyers and sellers on that circle. 

    input:
    -----

    a_seller_pos:       Array of seller locations (values between 0 and 2pi)
    a_buyer_pos:        Array of buyer locations (values between 0 and 2pi)
    a_color:            Array of length a_seller_pos, with one color per seller 
    seller_ind:         If not None, should be for each buyer the index number of the seller they bought from the most. 
                        This is then used to color the buyers in the same color as the respective sellers. If this 
                        variable is set to None, all buyers will be plotted in grey. 
    ax:                 Axis instance to plot into. 
    """

    # create the plot window and define other plotting related quantities
    ####################################################################################################################
    if ax is None:

        fig = plt.figure(figsize=(7,7))
        ax  = plt.gca()

    if a_color is None: a_color = sb.color_palette("deep",2)

    # plot a grey circle 
    ####################################################################################################################    
    r       = 1
    angles  = np.linspace(0, 2*np.pi, 500)
    xs      = r * np.sin(angles)                                    
    ys      = r * np.cos(angles) 
    ax.plot(xs, ys, color='grey', linestyle='--', zorder=1)

    # plot the seller locations 
    ####################################################################################################################    
    for i, s in enumerate(a_seller_pos):                        
        xcoord = [ r * np.sin( 2*np.pi*s ) ]
        ycoord = [ r * np.cos( 2*np.pi*s ) ]
        ax.scatter(xcoord, ycoord, marker='o', facecolor=a_color[i], alpha=0.8, 
                color=a_color[i], s=400, zorder=3, label='Firm %d'%(i+1))

    # for each buyer, find index of his seller
    ####################################################################################################################
    for j, b in enumerate(a_buyer_pos): 

        col    = 'ForestGreen' if seller_ind is None else a_color[seller_ind[j]]
        label  = ''            if seller_ind is None else 'Buyer who bought from Firm %d'%(seller_ind[j]+1)
        xcoord = [ r * np.sin( 2*np.pi*b ) ]
        ycoord = [ r * np.cos( 2*np.pi*b ) ]        

        ax.scatter(xcoord, ycoord, marker='x', color=col, s=150, zorder=2, label=label ) 

    # adjust the axis limits
    ####################################################################################################################
    ax.set_xlim(-1.2*r, 1.2*r)
    ax.set_ylim(-1.2*r, 1.5*r)
    ax.legend(loc='center', ncol=2, frameon=False, fontsize=8, labelspacing=2)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axis('equal')
