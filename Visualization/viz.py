import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import snsStyle

def addAxisPlotBuffer(ax, x, y):
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    xMargin = (xLim[1]-xLim[0])*x
    yMargin = (yLim[1]-yLim[0])*y

    ax.set_xlim(xLim[0] - xMargin, xLim[1] + xMargin)
    ax.set_ylim(yLim[0] - yMargin, yLim[1] + yMargin)

def vizSns2dScatter(df, x, y, yShift, title, chartSize = 15, yDollars = False):
    g = sns.lmplot(x = x
                  ,y = y
                  ,data = df
                  ,fit_reg = False
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
    xMin = 0 if 1.00 >= xChange >= 0.75 else np.round(xMin,1) #- xMin * 0.1
    xMax = xMax + xMax * 0.1

    yMin = round(np.min(df[y]), 5); yMax = round(np.max(df[y]), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= 0.75 else  np.round(yMin,1) #- yMin * 0.1
    yMax = yMax + yMax * 0.1
    plt.axis([xMin, xMax, yMin, yMax])    
    
    if yDollars:
        fmt = '${x:,.0f}'    # dollar sign formatting
        tick = tkr.StrMethodFormatter(fmt)
        g.ax.yaxis.set_major_formatter(tick)
        #g.yaxis.set_major_formatter(tick)
        

    plt.tight_layout()

    addAxisPlotBuffer(ax = g.ax, x = 0.01, y = 0.01) 
    return g.ax

def vizSns2dHist(x, ylabel, yShift, bins = 20, kde = False, rug = False, chartSize = 15, yDollars = False):
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

def vizSnsHeatmap(df, cols, chartSize):
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