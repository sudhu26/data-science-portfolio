import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.lines as Line2D

import snsStyle
import vizUtil

def viz2dScatter(df, x, y, targetCol = None, targetLabels = None, chartSize = 15, colorOptions = vizUtil.colorOptions, yShift = 0.8, title = '', xUnits = None, yUnits = None):
    # Create plotting objects
    fig, ax = plt.subplots()
    fig.set_size_inches(chartSize, chartSize * 0.6); plt.tight_layout()
    
    # 2d scatter, one color
    if targetCol is None:
        X = df[[x,y]].values
        plt.scatter(x = X[:,0]
                        ,y = X[:,1]
                        ,color = snsStyle.colTitleGrey
                        #,label = targetLabel
                        ,s = 150
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 2.5
                    )
    else:
        # 2d scatter with hue
        # Transform pandas dataframe to numpy array for visualization    
        X = df[[x,y,targetCol]].values
        targetIds =  np.unique(X[:, 2])        
        
        # Plot data points
        for targetId, targetLabel, color in zip(targetIds, targetLabels, colorOptions[:len(targetIds)]):
            plt.scatter(x = X[X[:,2] == targetId][:,0]
                        ,y = X[X[:,2] == targetId][:,1]
                        ,color = color
                        ,label = targetLabel
                        ,s = 150
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 3.5
                    )
        lgd = plt.legend(loc = 'right'
                ,bbox_to_anchor = (0., 1.5, 0.9, -1.302)
                ,ncol = 1
                ,borderaxespad = -11.5
                ,frameon = True
                ,fontsize = 20)
    # Text labels
    ax.set_title(title, fontsize = 30, color = snsStyle.colTitleGrey, loc = 'left', pad = 25)
    plt.xlabel(x, fontsize = 25, labelpad = 25, position = (0.5, 0.5))
    plt.ylabel(y, fontsize = 25, labelpad = 25, position = (1.0, yShift))
    vizUtil.vizUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 20, yUnits = yUnits, ySize = 20)
    
    # Dynamically set axis lower / upper limits
    xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(df = df, x = x, y = y)
    plt.axis([xMin, xMax, yMin, yMax])   
    
    vizUtil.vizUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)
    #plt.show()
    return fig, ax

def viz2dHist(x, ylabel, yShift, bins = 20, kde = False, rug = False, chartSize = 15, yDollars = False):
    fig = plt.subplots(figsize = (chartSize, chartSize * 0.6))
    g = sns.distplot(x
                    ,bins =bins
                    ,kde = kde
                    ,rug = rug
                    ,color = snsStyle.colTitleGrey)
    g.set_xlabel(xlabel = '')
    g.set_ylabel(ylabel = ylabel, labelpad = 25, position = (1.0, yShift))    

    if yDollars:
        fmt = '${x:,.0f}'    # dollar sign formatting
        tick = tkr.StrMethodFormatter(fmt)
        g.yaxis.set_major_formatter(tick)   
    g.set_xticklabels('')
    plt.xticks([])
    plt.tight_layout()
    return g

def vizCorrHeatmap(df, cols, chartSize):
    fig = plt.subplots(figsize = (chartSize, chartSize))
    corrMatrix = df.corr()
    corrMatrix = corrMatrix.loc[cols][cols]

    mask = np.zeros_like(corrMatrix)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corrMatrix
               ,cmap = 'Greys'
               ,mask = mask
               ,square = True)

def func():
    pass