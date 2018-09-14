import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.patches as mpatches
import matplotlib.lines as Line2D

import snsStyle
import vizUtil

def viz2dScatter(df, x, y, yShift = 0.5, title = '', chartSize = 15, yDollars = False):
    
    g = sns.lmplot(x = x
                  ,y = y
                  ,data = df
                  ,fit_reg = False
                  ,legend = False
                  ,scatter_kws = {'color' : snsStyle.colWhite
                                  ,'s' : 200
                                  ,'alpha' : 0.7
                                  ,'linewidths' : 2.5
                                  ,'edgecolor' : snsStyle.colTitleGrey
                                  } 
                    )
    g.ax.set_title(title, color = snsStyle.colTitleGrey, loc = 'left', pad = 50)
    
    g.ax.set_xlabel(xlabel = x, labelpad = 25, position = (0.5, 0.5))
    g.ax.set_ylabel(ylabel = y, labelpad = 25, position = (1.0, yShift))
    g.fig.set_size_inches(chartSize, chartSize * 0.6)
    
    xMin = round(np.min(df[x]), 5); xMax = round(np.max(df[x]), 5)
    xChange = (xMax - xMin) / xMax
    xMin = 0 if 1.00 >= xChange >= 0.75 else np.round(xMin,1)
    xMax = xMax + xMax * 0.1

    yMin = round(np.min(df[y]), 5); yMax = round(np.max(df[y]), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= 0.75 else  np.round(yMin,1)
    yMax = yMax + yMax * 0.1
    plt.axis([xMin, xMax, yMin, yMax])    
    
    if yDollars:
        fmt = '${x:,.0f}'    # dollar sign formatting
        tick = tkr.StrMethodFormatter(fmt)
        g.ax.yaxis.set_major_formatter(tick)
    plt.tight_layout()
    addAxisPlotBuffer(ax = g.ax, x = 0.01, y = 0.01) 
    return g.ax

def viz2dScatterHue(df, x, y, targetCol, targetLabels, chartSize = 15, colorOptions = vizUtil.colorOptions, yShift = 0.8, title = '', xUnits = None, yUnits = None):
    # Transform pandas dataframe to numpy array for visualization    
    X = df[[x,y,targetCol]].values
    targetIds =  np.unique(X[:,2])
    
    # Create plotting objects
    fig, ax = plt.subplots()
    fig.set_size_inches(chartSize, chartSize * 0.6); plt.tight_layout()
    
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
    
    # Text labels
    ax.set_title(title, fontsize = 30, color = snsStyle.colTitleGrey, loc = 'left', pad = 25)
    plt.xlabel(x, fontsize = 25, labelpad = 25, position = (0.5, 0.5))
    plt.ylabel(y, fontsize = 25, labelpad = 25, position = (1.0, yShift))
    vizUtil.labelFormatter(ax = ax, xUnits = xUnits, xSize = 20, yUnits = yUnits, ySize = 20)
    lgd = plt.legend(loc = 'right'
                    ,bbox_to_anchor = (0., 1.5, 0.9, -1.302)
                    ,ncol = 1
                    ,borderaxespad = -11.5
                    ,frameon = True
                    ,fontsize = 20)
    
    # Dynamically set axis lower / upper limits
    xMin = round(np.min(df[x]), 5); xMax = round(np.max(df[x]), 5)
    xChange = (xMax - xMin) / xMax
    xMin = 0 if 1.00 >= xChange >= 0.75 else np.round(xMin,1)
    xMax = xMax + xMax * 0.1
    yMin = round(np.min(df[y]), 5); yMax = round(np.max(df[y]), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= 0.75 else  np.round(yMin,1)
    yMax = yMax + yMax * 0.1
    plt.axis([xMin, xMax, yMin, yMax])  
    
    vizUtil.plotAreaBuffer(ax = ax, x = 0.02, y = 0.02)
    plt.show()
    return ax

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