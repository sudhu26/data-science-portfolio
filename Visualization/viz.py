import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.lines as Line2D

import vizStyle
import vizUtil

class QuickPlot:

    #plt.rcParams['figure.figsize'] = [15, 10]
        
    def __init__(self, fig = plt.figure(), chartProp = 15, orientation = None):
        """
        
        """
        self.chartProp = chartProp
        self.orientation = orientation
        self.fig = fig
                
        chartWidth = self.chartProp * .4 if self.orientation == 'tall' else self.chartProp
        chartHeight = self.chartProp * .6 if self.orientation == 'tall' else self.chartProp * .5
        
        self.fig.set_figheight(chartHeight)
        self.fig.set_figwidth(chartWidth)
        
        #plt.rcParams['figure.figsize'] = [chartWidth, chartHeight]
        
    def makeCanvas(self, title = '', xLabel = '', yLabel = '', yShift = 0.8, position = 111):
        """
        Add basic informational components to figure, including titles and axis labels
        """        
        ax = self.fig.add_subplot(position)
        
        # Set additional defaults for plot figure, including chart size and layout
        # Add title
        ax.set_title(title, fontsize = 1.999 * self.chartProp, color = vizStyle.vizGrey, loc = 'left', pad = 1.667 * self.chartProp)
    
        # Add axis labels
        plt.xlabel(xLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (0.5, 0.5))
        plt.ylabel(yLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (1.0, yShift))
        return ax

    def viz2dScatter(self, x, y, df = None, xUnits = None, yUnits = None, ax = None):
        """
        
        """
        if df is not None:
            x = df[x].values
            y = df[y].values
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(x = x, y = y)
        else:
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(x = x, y = y)

        # 2d scatter, one color
        plt.scatter(x = x
                    ,y = y
                    ,color = vizStyle.vizGrey
                    ,s = 10 * self.chartProp
                    ,alpha = 0.7
                    ,facecolor = 'w'
                    ,linewidth = 0.167 * self.chartProp
                   )
        
        # Text labels
        vizUtil.vizUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)        
    
        # Dynamically set axis lower / upper limits
        plt.axis([xMin, xMax, yMin, yMax])   
        vizUtil.vizUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        plt.tight_layout()
    
    def viz2dScatterHue(self, x, y, df = None, targetCol = None, targetLabels = None, xUnits = None, yUnits = None):
        """
        
        """
        if df is not None:
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(df = df, x = x, y = y)
            x = df[x].values
            y = df[y].values
        else:
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(x = x, y = y)

        # 2d scatter with hue
        # Transform pandas dataframe to numpy array for visualization    
        X = df[[x, y, targetCol]].values
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
        vizUtil.vizUtilLabelFormatter(ax = self.ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)
    
        # Dynamically set axis lower / upper limits
        plt.axis([xMin, xMax, yMin, yMax])   
        vizUtil.vizUtilPlotBuffer(ax = self.ax, x = 0.02, y = 0.02)

    def vizLine(self, x, y, colNames = None, df = None, linecolor = vizStyle.vizColors[0], linestyle = vizStyle.vizLineStyle[0], multiValAxis = 'x', xUnits = None, yUnits = None, markerOn = False, ax = None):
        """
        Parameters:
            df = Pandas DataFrame. Select columns get converted to numpy arrays.
            xColNames = list, column names to chart x-axis value.
            yColNames = list, column names to chart y-axis value.
            multiValAxis = str, dictates whether the 'x' axis or the 'y' axis plots different values for each series
            targetLabels (optional) = list, contains category labels of unique values in targetCol. Used for creating legend labels.
            chartProp (optional) = Float. Controls proportionality of chart, ticklabels, tick marks, axis labels and title.
            yShift (optional) = Float. Position y-axis label up/down axis.
            title (optional) = str, chart title.
            xUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
            yUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
            orientation = str. None provides wide orientation, 'tall' provides tall orientation.
        """
        if df is not None:
            X = df[x].values
            y = df[y].values
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(x = X, y = y)
        else:
            X = x
            xMin, xMax, yMin, yMax = vizUtil.vizUtilSetAxes(x = X, y = y)

        if multiValAxis == 'x':
            for ix in np.arange(X.shape[1]):
                xCol = X[:, ix]
                plt.plot(xCol
                         ,y
                         ,color = linecolor
                         ,linestyle = vizStyle.vizLineStyle[ix]
                         ,linewidth = 0.167 * self.chartProp
                         ,label = colNames[ix] if colNames is not None else None
                         ,marker = '.' if markerOn else None
                         ,markersize = 25 if markerOn else None
                         ,markerfacecolor = 'w' if markerOn else None
                         ,markeredgewidth = 2.5 if markerOn else None
                        )                
        else:
            for ix in np.arange(y.shape[1]):
                yCol = y[:, ix]
                plt.plot(x
                         ,yCol
                         ,color = vizStyle.vizColors[ix]
                         ,linestyle = vizStyle.vizLineStyle[ix]
                         ,linewidth = 0.167 * self.chartProp
                         ,label = colNames[ix]
                         ,marker = '.' if markerOn else None
                         ,markersize = 25 if markerOn else None
                         ,markerfacecolor = 'w' if markerOn else None
                         ,markeredgewidth = 2.5 if markerOn else None
                        )

        if colNames is not None:
            plt.legend(loc = 'right'
                       ,bbox_to_anchor = (0., 1.5, 0.9, -1.302)
                       ,ncol = 1
                       ,borderaxespad = -0.766 * self.chartProp
                       ,frameon = True
                       ,fontsize = 1.333 * self.chartProp
                      )
        
        # Text labels
        vizUtil.vizUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)
    
        # Dynamically set axis lower / upper limits
        plt.axis([xMin, xMax, yMin, yMax])   
        vizUtil.vizUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        plt.tight_layout()
            
def viz2dScatter(df = None, x = None, y = None, targetCol = None, targetLabels = None, yShift = 0.8, title = '', xUnits = None, yUnits = None, orientation = None, ax = None):
    """
    Creates 2-dimensional scatter plot. If targetCol is set to None, scatterplot is monochromatic. If targetCol is set to a specific column, 
    scatter plot takes on a hue based on the unique categories in targetCol

    Parameters:
        df = Pandas DataFrame. Select columns get converted to numpy arrays.
        xColName = str, column name of DataFrame. x-axis value.
        yColName = str, column name of DataFrame. y-axis value.
        targetCol (optional) = str, column name of DataFrame. Column used for differentiating x and y values by color.
        targetLabels (optional) = list, contains category labels of unique values in targetCol. Used for creating legend labels.
        chartProp (optional) = Float. Controls proportionality of chart, ticklabels, tick marks, axis labels and title.
        yShift (optional) = Float. Position y-axis label up/down axis.
        title (optional) = str, chart title.
        xUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        yUnits (optional) = str. '$' displays tick labels in dollars, '%' displays tick labels as percentages.
        orientation = str. None provides wide orientation, 'tall' provides tall orientation.
    """    
    
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

