import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

#import snsStyle

colorOptions = ['#FF4F00' # orange
               ,'#7F5EBA' # purple
               ,'#FFC02E' # yellow
               ,'#9BD020' # green
               ,'#01ADAD'] # teal 

def vizUtilPlotBuffer(ax, x, y):
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    xMargin = (xLim[1]-xLim[0])*x
    yMargin = (yLim[1]-yLim[0])*y

    ax.set_xlim(xLim[0] - xMargin, xLim[1] + xMargin)
    ax.set_ylim(yLim[0] - yMargin, yLim[1] + yMargin)

def vizUtilLabelFormatter(ax, xUnits, xSize, yUnits, ySize):
    # x-axis
    if xUnits == '$':
        fmt = '${x:,.0f}'
    elif xUnits == '%':
        fmt = '{x:,.1f}%'
    else:
        fmt = '{x:,.1f}'    
    tick = tkr.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick)

    for tk in ax.get_xticklabels():
        tk.set_fontsize(20)

    # y-axis
    if yUnits == '$':
        fmt = '${x:,.0f}'
    elif yUnits == '%':
        fmt = '{x:,.1f}%'
    else:
        fmt = '{x:,.1f}'    
    tick = tkr.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    
    for tk in ax.get_yticklabels():
        tk.set_fontsize(20)

def vizUtilSetAxes(df, x, y, xThresh = 0.75, yThresh = 0.75):
    xMin = round(np.min(df[x]), 5); xMax = round(np.max(df[x]), 5)
    xChange = (xMax - xMin) / xMax
    xMin = 0 if 1.00 >= xChange >= xThresh else np.round(xMin,1)
    xMax = xMax + xMax * 0.1

    yMin = round(np.min(df[y]), 5); yMax = round(np.max(df[y]), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= yThresh else  np.round(yMin,1)
    yMax = yMax + yMax * 0.1
    return xMin, xMax, yMin, yMax

