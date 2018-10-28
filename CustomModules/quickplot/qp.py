import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap

import quickplot.qpStyle as qpStyle
import quickplot.qpUtil as qpUtil

class QuickPlot:

        
    def __init__(self, fig = plt.figure(), chartProp = 15, plotOrientation = None):
        """
        Info:
            Description: 
                Initialize QuickPlot, create figure and determine chart proportions, orientation.        
            Parameters:
                fig : figure object, default = plt.figure()
                    matplotlib.pyplot figure object is a top level container for all plot elements.
                chartProp : float or int, default = 15
                    Chart proportionality control. Determines relative size of figure size, axis labels, 
                    chart title, tick labels, tick marks.
                orientation : string, default = None
                    Default value produces a plot that is wider than it is tall. Specifying 'tall' will 
                    produce a taller, less wide plot.
        """
        self.chartProp = chartProp
        self.plotOrientation = plotOrientation
        self.fig = fig
        
        # Dynamically set chart width and height parameters
        chartWidth = self.chartProp * .7 if self.plotOrientation == 'tall' else self.chartProp
        chartHeight = self.chartProp * .8 if self.plotOrientation == 'tall' else self.chartProp * .5
        self.fig.set_figheight(chartHeight)
        self.fig.set_figwidth(chartWidth)

    def makeCanvas(self, title = '', xLabel = '', yLabel = '', yShift = 0.8, position = 111):
        """
        Info:
            Description: 
                Create Axes object, add descriptive attributes such as titles and labels.        
            Parameters:
                title : string, default = '' (blank)
                    The title for the chart.
                xLabel : string, default = '' (blank)
                    x-axis label.
                yLabel : string, default = '' (blank)
                    y-axis label.
                yShift : float, default = 0.8
                    Controls position of y-axis label. Higher values move label higher along axis. 
                    Intent is to align with top of axis.
                position int (nrows, ncols, index) : default = 111
                    Determine subplot position of plot.

            Returns 
                ax : Axes object
                    Contain figure elements
        """        
        ax = self.fig.add_subplot(position)
        
        # Add title
        ax.set_title(title, fontsize = 1.999 * self.chartProp, color = qpStyle.qpGrey, loc = 'left', pad = 1.667 * self.chartProp)
    
        # Add axis labels
        plt.xlabel(xLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (0.5, 0.5))
        plt.ylabel(yLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (1.0, yShift))
        return ax

    def qp2dScatter(self, x, y, df = None, xUnits = 'f', yUnits = 'f', plotBuffer = True
                    , axisLimits = True, ax = None):
        """
        Info:
            Description: 
                Create 2-dimensional scatter plot.
            Parameters:
                x : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                y : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                df : Pandas DataFrame, default=  None
                    Dataset containing data to be plotted. Can be any size, as plotted columns will be 
                    chosen by columns names specified in x, y. 
                xUnits : string, default = None
                    Determines units of x-axis tick labels. None displays float. '%' displays percentages, 
                    '$' displays dollars. 
                yUnits : string, default = None
                    Determines units of y-axis tick labels. None displays float. '%' displays percentages, 
                    '$' displays dollars.
                plotBuffer : boolean, default = True
                    Switch for determining whether dynamic plot buffer function is executed.
                axisLimits : boolean, default = True
                    Switch for determining whether dynamic axis limit setting function is executed.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        
        # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
        if df is not None:
            x = df[x].values
            y = df[y].values
                
        # 2d scatter
        plt.scatter(x = x
                    ,y = y
                    ,color = qpStyle.qpGrey
                    ,s = 10 * self.chartProp
                    ,alpha = 0.7
                    ,facecolor = 'w'
                    ,linewidth = 0.167 * self.chartProp
                   )
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)        
    
        # Dynamically set axis lower / upper limits
        if axisLimits:
            xMin, xMax, yMin, yMax = qpUtil.qpUtilSetAxes(x = x, y = y)        
            plt.axis([xMin, xMax, yMin, yMax])   

        # Create smaller buffer around plot area to prevent cutting off elements.
        if plotBuffer:
            qpUtil.qpUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        # Show figure with tight layout.
        plt.tight_layout()
    
    def qp2dScatterHue(self, x, y, target, label, df = None, xUnits = 'd', yUnits = 'd', plotBuffer = True
                        , axisLimits = True, bbox = (1.2, 0.9), ax = None):
        """
        Info:
            Description:

            Parameters:

        """
        if df is not None:
            X = df[[x, y, target]].values
            x = df[x].values
            y = df[y].values
            target = df[target].values
        else:
            X = np.c_[x, y, target]

        targetIds =  np.unique(X[:, 2])
            
        # 2d scatter with hue
        # Transform pandas dataframe to numpy array for visualization    
        
        # Plot data points
        for targetId, targetLabel, color in zip(targetIds, label, qpStyle.qpColorsHexMid[:len(targetIds)]):
            plt.scatter(x = X[X[:,2] == targetId][:,0]
                        ,y = X[X[:,2] == targetId][:,1]
                        ,color = color
                        ,label = targetLabel
                        ,s = 10 * self.chartProp
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 0.234 * self.chartProp
                    )
        # Add legend to figure
        if label is not None:
            plt.legend(loc = 'upper right'
                       ,bbox_to_anchor = bbox
                       ,ncol = 1
                       ,frameon = True
                       ,fontsize = 1.1 * self.chartProp
                      )
            
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)
    
        # Dynamically set axis lower / upper limits
        if axisLimits:
            xMin, xMax, yMin, yMax = qpUtil.qpUtilSetAxes(x = x, y = y)
            plt.axis([xMin, xMax, yMin, yMax])   
        
        # Create smaller buffer around plot area to prevent cutting off elements.
        if plotBuffer:
            qpUtil.qpUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        # Show figure with tight layout.
        plt.tight_layout()

    def qpLine(self, x, y, label = None, df = None, linecolor = None, linestyle = None
                , bbox = (1.2, 0.9), yMultiVal = False, xUnits = 'f', yUnits = 'f', markerOn = False
                , plotBuffer = False, axisLimits = False, ax = None):
        """
        Info:
            Description: 
                Create 2-dimensional line plot. Capable of plotting a single line, multi lines on a single axis, 
                or multiple lines on multiple axes.

            Parameters:
                x : array or string
                    Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                    in a Pandas DataFrame, or a column name in a Pandas DataFrame.
                y : array or string
                    Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                    in a Pandas DataFrame, or a column name in a Pandas DataFrame.
                label : list of strings : default = None
                    List of names of used to create legend entries for each line.
                df : Pandas DataFrame, default = None
                    Dataset containing data to be plotted. Can be any size, as plotted columns will be chosen 
                    by columns names specified in x, y. 
                linecolor : string, default = reference to list
                    Determine color of line.
                linestyle : string, default = reference to list
                    Determine style of line.
                bbox : tuple, default = (1.2, 0.9)
                    Override bbox value for legend
                yMultiVal : boolean : default = False
                    If a single x value is paired with multiple y values, set to True.
                xUnits : string, default = 'd'
                    Determines units of x-axis tick labels. None displays float. '%' displays percentages, 
                    '$' displays dollars. 
                yUnits : string, default = 'd'
                    Determines units of y-axis tick labels. None displays float. '%' displays percentages, 
                    '$' displays dollars.
                markerOn : boolean, default = False
                    Determines whether to show line with markers at each element.
                plotBuffer : boolean, default = False
                    Switch for determining whether dynamic plot buffer function is executed.
                axisLimits : boolean, default = False
                    Switch for determining whether dynamic axis limit setting function is executed.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within `function.
        """
        
        # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
        if df is not None:
            x = df[x].values
            y = df[y].values
        
        # Add line 
        if not yMultiVal:
            for ix in np.arange(x.shape[1]):
                xCol = x[:, ix]
                plt.plot(xCol
                         ,y
                         ,color = linecolor if linecolor is not None else qpStyle.qpColorsHexMid[ix]
                         ,linestyle = linestyle if linestyle is not None else qpStyle.qpLineStyle[0]
                         ,linewidth = 0.167 * self.chartProp
                         ,label = label[ix] if label is not None else None
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
                         ,color = linecolor if linecolor is not None else qpStyle.qpColorsHexMid[ix]
                         ,linestyle = linestyle if linestyle is not None else qpStyle.qpLineStyle[0]
                         ,linewidth = 0.167 * self.chartProp
                         ,label = label[ix] if label is not None else None
                         ,marker = '.' if markerOn else None
                         ,markersize = 25 if markerOn else None
                         ,markerfacecolor = 'w' if markerOn else None
                         ,markeredgewidth = 2.5 if markerOn else None
                        )

        # Add legend to figure
        if label is not None:
            plt.legend(loc = 'upper right'
                       ,bbox_to_anchor = bbox
                       ,ncol = 1
                       ,frameon = True
                       ,fontsize = 1.1 * self.chartProp
                      )
            
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)
    
        # Dynamically set axis lower / upper limits
        if axisLimits:
            xMin, xMax, yMin, yMax = qpUtil.qpUtilSetAxes(x = x, y = y)
            plt.axis([xMin, xMax, yMin, yMax])   
        
        # Create smaller buffer around plot area to prevent cutting off elements.
        if plotBuffer:
            qpUtil.qpUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        # Show figure with tight layout.
        plt.tight_layout()    
    
    def qpHist(self, x, y, xLabels, labelRotate = 0, log = False, orientation = 'vertical'
                 , yUnits = 'ff', ax = None):
        """
        Info:
            Description:

            Parameters:
                x : Array
                y : Array
                xLabels : List
                log : boolean, default = True
        """
        plt.bar(x = x
                ,height = y
                ,color = qpStyle.qpGrey
                ,tick_label = xLabels
                ,orientation = orientation
                ,alpha = 0.5
            )

        plt.xticks(rotation = labelRotate)

        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xSize = 1.333 * self.chartProp, yUnits = yUnits, ySize = 1.333 * self.chartProp)    

    def qpCorrHeatmap(self, df, cols, chartProp):
        """
        Info:
            Description:

            Parameters:

        """
        fig = plt.subplots(figsize = (chartProp, chartProp))
        corrMatrix = df.corr()
        corrMatrix = corrMatrix.loc[cols][cols]

        mask = np.zeros_like(corrMatrix)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(corrMatrix
                ,cmap = 'Greys'
                ,mask = mask
                ,square = True)

    def qpDecisionRegion(self, x, y, classifier, testIdx = None, resolution = 0.001, bbox = (1.2, 0.9), ax = None):
        """
        Info:
            Description:
                Create 2-dimensional chart with shading used to highlight decision regions
            Parameters:
                X : Array
                    m x 2 array containing 2 features
                y : Array
                    m x 1 array containing labels for observations
                classifier : sklearn model
                    Classifier used to create decision regions
                testIdx :  tuple, default = None
                    Option parameter for specifying observations to be highlighted
                    as test examples
                resolution : float, default = 0.001
                    Controls clarity of the graph by setting interval of the arrays 
                    passed into np.meshgrid
        """
        # objects for marker generator and color map
        cmap = ListedColormap(qpStyle.qpColorsHexLight[:len(np.unique(y))])
        
        # plot decision surface
        x1Min, x1Max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x2Min, x2Max = x[:, 1].min() - 1, x[:, 1].max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1Min, x1Max, resolution)
                            ,np.arange(x2Min, x2Max, resolution))
        
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, slpha = 0.3, cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        # Plot samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x = x[y == cl, 0]
                    ,y = x[y == cl, 1]
                    ,alpha = 1.0
                    ,c = qpStyle.qpColorsHexMid[idx]
                    ,marker = qpStyle.qpMarkers[1]
                    ,label = cl
                    ,s = 12.5 * self.chartProp
                    ,edgecolor = qpStyle.qpColorsHexMidDark[idx]
                    )
        
        # Highlight test samples
        if testIdx:
            xTest = x[testIdx, :]
            plt.scatter(xTest[:,0]
                        ,xTest[:,1]
                        ,facecolor = 'none'
                        ,edgecolor = 'white'
                        ,alpha = 1.0
                        ,linewidth = 1.4
                        ,marker = 'o'
                        ,s = 12.75 * self.chartProp
                        ,label = 'test set'                   
                    )
        # Add legend to figure
        plt.legend(loc = 'upper right'
                    ,bbox_to_anchor = bbox
                    ,ncol = 1
                    ,frameon = True
                    ,fontsize = 1.1 * self.chartProp
                    )
        plt.tight_layout()
