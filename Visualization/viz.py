import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.lines as Line2D

import vizStyle
import vizUtil

def viz2dScatter(df, x, y, targetCol = None, targetLabels = None, chartProp = 15, yShift = 0.8, title = '', xUnits = None, yUnits = None, orientation = None, ax = None):
    """
    Creates 2-dimensional scatter plot. If targetCol is set to None, scatterplot is monochromatic. If targetCol is set to a specific column, 
    scatter plot takes on a hue based on the unique categories in targetCol

    Parameters:
        df = Pandas DataFrame. Select columns get converted to numpy arrays.
        x = str, column name of DataFrame. x-axis value.
        y = str, column name of DataFrame. y-axis value.
        targetCol (optional) = str, column name of DataFrame. Column used for differentiating x and y values by color.
        targetLabels (optional) = list, contains category labels of unique values in targetCol. Used for creating legend labels.
        chartProp (optional) = Float. Controls proportionality of chart, ticklabels, tick marks, axis labels and title.
        yShift (optional) = Float. Position y-axis label up/down axis.
        title (optional) = str, chart title.
        xUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        yUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        orientation = str. None provides wide orientation, 'tall' provides tall orientation.
    """    
    # Create plotting objects

    chartWidth = chartProp * .4 if orientation == 'tall' else chartProp
    chartHeight = chartProp * .6 if orientation == 'tall' else chartProp * .5

    fig, ax = plt.subplots()
    fig.set_size_inches(chartWidth, chartHeight); plt.tight_layout()

    # 2d scatter, one color
    if targetCol is None:
        X = df[[x,y]].values
        plt.scatter(x = X[:,0]
                        ,y = X[:,1]
                        ,color = vizStyle.vizGrey
                        ,s = 10 * chartProp
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 0.167 * chartProp
                    )
    else:
        # 2d scatter with hue
        # Transform pandas dataframe to numpy array for visualization    
        X = df[[x,y,targetCol]].values
        targetIds =  np.unique(X[:, 2])        
        
        # Plot data points
        for targetId, targetLabel, color in zip(targetIds, targetLabels, vizStyle.vizColors[:len(targetIds)]):
            plt.scatter(x = X[X[:,2] == targetId][:,0]
                        ,y = X[X[:,2] == targetId][:,1]
                        ,color = color
                        ,label = targetLabel
                        ,s = 10 * chartProp
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 0.234 * chartProp
                    )
        lgd = plt.legend(loc = 'right'
                ,bbox_to_anchor = (0., 1.5, 0.9, -1.302)
                ,ncol = 1
                ,borderaxespad = -0.766 * chartProp
                ,frameon = True
                ,fontsize = 1.333 * chartProp)
    # Text labels
    ax.set_title(title, fontsize = 1.999 * chartProp, color = vizStyle.vizGrey, loc = 'left', pad = 1.667 * chartProp)
    plt.xlabel(x, fontsize = 1.667 * chartProp, labelpad = 1.667 * chartProp, position = (0.5, 0.5))
    plt.ylabel(y, fontsize = 1.667 * chartProp, labelpad = 1.667 * chartProp, position = (1.0, yShift))
    vizUtil.vizUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * chartProp, yUnits = yUnits, ySize = 1.333 * chartProp)
    
    # Dynamically set axis lower / upper limits
    xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(df = df, x = x, y = y)
    plt.axis([xMin, xMax, yMin, yMax])   
    
    vizUtil.vizUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)
    return fig, ax

def vizLine(df, xCols, y, targetLabels = None, chartProp = 15, yShift = 0.8, title = '', xUnits = None, yUnits = None, orientation = None, ax = None):
    """
    Parameters:
        df = Pandas DataFrame. Select columns get converted to numpy arrays.
        xCols = list, column namejs to chart x-axis value.
        y = str, column name of DataFrame. y-axis value.
        targetLabels (optional) = list, contains category labels of unique values in targetCol. Used for creating legend labels.
        chartProp (optional) = Float. Controls proportionality of chart, ticklabels, tick marks, axis labels and title.
        yShift (optional) = Float. Position y-axis label up/down axis.
        title (optional) = str, chart title.
        xUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        yUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        orientation = str. None provides wide orientation, 'tall' provides tall orientation.
    
    """
    chartWidth = chartProp * .4 if orientation == 'tall' else chartProp
    chartHeight = chartProp * .6 if orientation == 'tall' else chartProp * .5

    fig, ax = plt.subplots()
    
    fig.set_size_inches(chartWidth, chartHeight); plt.tight_layout()

    # 2d line
    X = df[xCols].values
    y = df[y].values

    for colIx in np.arange(len(xCols)):
        x = X[:,colIx]
        plt.plot(x
                 ,y
                 ,color = vizStyle.vizColors[colIx]
                 ,linestyle = vizStyle.vizLineStyle[colIx]
                 ,linewidth = 0.167 * chartProp
                 ,label = xCols[colIx]
                 )
        lgd = plt.legend(loc = 'right'
                ,bbox_to_anchor = (0., 1.5, 0.9, -1.302)
                ,ncol = 1
                ,borderaxespad = -0.766 * chartProp
                ,frameon = True
                ,fontsize = 1.333 * chartProp)
    # Text labels
    ax.set_title(title, fontsize = 1.999 * chartProp, color = vizStyle.vizGrey, loc = 'left', pad = 1.667 * chartProp)
    vizUtil.vizUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * chartProp, yUnits = yUnits, ySize = 1.333 * chartProp)
    
    vizUtil.vizUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)
    return fig, ax

def viz2dHist(x, ylabel, yShift, bins = 20, kde = False, rug = False, chartProp = 15, yDollars = False):
    pass

def vizCorrHeatmap(df, cols, chartProp):
    fig = plt.subplots(figsize = (chartProp, chartProp))
    corrMatrix = df.corr()
    corrMatrix = corrMatrix.loc[cols][cols]

    mask = np.zeros_like(corrMatrix)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corrMatrix
               ,cmap = 'Greys'
               ,mask = mask
               ,square = True)

