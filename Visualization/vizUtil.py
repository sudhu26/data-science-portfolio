import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

def vizUtilPlotBuffer(ax, x, y):
    """
    Creates narrow border around plot arrow. 
    Prevents plot icons from being clipped by plot edges.
    """
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    xMargin = (xLim[1] - xLim[0]) * x
    yMargin = (yLim[1] - yLim[0]) * y

    ax.set_xlim(xLim[0] - xMargin, xLim[1] + xMargin)
    ax.set_ylim(yLim[0] - yMargin, yLim[1] + yMargin)

def vizUtilLabelFormatter(ax, xUnits = None, xSize = None, yUnits = None, ySize = None):
    """
    Info:
        Description:
            Formats tick labels as dolloars, percentages, or decimals.
        Parameters:
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
            xUnits : str, default = None
                Determines units of x-axis tick labels. None displays float. '%' displays percentages, 
                '$' displays dollars.
            xSize : int or float, default = None
                x-axis label size
            yUnits : str, default = None
                Determines units of y-axis tick labels. None displays float. '%' displays percentages, 
                '$' displays dollars.
            ySize : int or float, default = None
                y-axis label size
    """
    # x-axis
    if xUnits is not None:
        if xUnits == '$':
            fmt = '${x:,.0f}'
        elif xUnits == '%':
            fmt = '{x:,.1f}%'
        else:
            fmt = '{x:,.2f}'    
        tick = tkr.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)

    for tk in ax.get_xticklabels():
        tk.set_fontsize(xSize)

    # y-axis
    if yUnits is not None:
        if yUnits == '$':
            fmt = '${x:,.0f}'
        elif yUnits == '%':
            fmt = '{x:,.1f}%'
        else:
            fmt = '{x:,.2f}'    
        tick = tkr.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
        
    for tk in ax.get_yticklabels():
        tk.set_fontsize(ySize)

def vizUtilSetAxes(x, y, xThresh = 0.75, yThresh = 0.75):
    """
    Dynamically set lower/upper limits of x/y axes.
    """
    xMin = round(np.min(x), 5)
    xMax = round(np.max(x), 5)
    xChange = (xMax - xMin) / xMax
    xMin = 0 if 1.00 >= xChange >= xThresh else np.round(xMin,1)
    xMax = xMax + xMax * 0.1

    yMin = round(np.min(y), 5)
    yMax = round(np.max(y), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= yThresh else  np.round(yMin,1)
    yMax = yMax + yMax * 0.1
    return xMin, xMax, yMin, yMax

