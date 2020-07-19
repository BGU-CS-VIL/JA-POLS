
import matplotlib.pyplot as plt
from textwrap import wrap
# if we want to save fig, otherwise- remove this line:
plt.switch_backend('agg')


# =============================================================================
# Figures $ subplots
# =============================================================================

def open_figure(figure_num, figure_title, figsize):
    fig = plt.figure(figure_num, figsize)
    plt.clf()
    plt.suptitle(figure_title, fontsize=10)
    return fig
    

def PlotImages(figure_num, rows, cols, ind, images, titles, cmap, axis=True, colorbar=True, **kwargs):
    plt.figure(figure_num)
    #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=0.7, hspace=0.8)
    for i in range(len(images)):
        if(images[i] is None):
            continue
        ax = plt.subplot(rows, cols, i+ind)
        ax.set_title("\n".join(wrap(titles[i], 60)), fontsize= 9)
        _kwargs = {}
        if 'm' in kwargs:
            m = kwargs['m']
            _kwargs['vmin']=-m
            _kwargs['vmax']=m
        img = ax.imshow(images[i], cmap=cmap, interpolation = 'None', **_kwargs)
        if axis == False:
            plt.axis('off')
        if colorbar == True:
            plt.colorbar(img, fraction=0.046, pad=0.04)
