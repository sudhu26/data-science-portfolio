#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

#import snsStyle

colorOptions = ['#FF4F00' # orange
               ,'#7F5EBA' # purple
               ,'#FFC02E' # yellow
               ,'#9BD020' # green
               ,'#01ADAD'] # teal 

def plotAreaBuffer(ax, x, y):
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    xMargin = (xLim[1]-xLim[0])*x
    yMargin = (yLim[1]-yLim[0])*y

    ax.set_xlim(xLim[0] - xMargin, xLim[1] + xMargin)
    ax.set_ylim(yLim[0] - yMargin, yLim[1] + yMargin)

def labelFormatter(ax, xUnits, xSize, yUnits, ySize):
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
