


# nf          = 3
# xticks      = np.arange(nf)
# xticklabels = ['firm %s'i for i in np.arange(1,nf+1)]

# ax.set_xlabel('')




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

