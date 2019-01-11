import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sklearn.metrics as metrics
import sklearn.preprocessing as prepocessing

from statsmodels.stats.weightstats import ztest
from scipy import stats

import quickplot.qpStyle as qpStyle
import quickplot.qpUtil as qpUtil

from IPython.display import display_html
    
sns.set_style('whitegrid')

class QuickPlot:
    """
    Info:
        Description:
            QuickPlot is a class that contain methods for creating data visualization.
            Initialization of this class creates a plotting object of a chosen size and
            orientation. Once the figure is initialized, the method makeCanvas is called
            to create the figure axis or chosen number of axes. If multiple axes are 
            specified, then multiple axes can be plotted on a single figure, or the 
            position variable can be utilized to create a subplot arrangement.        
    """        
    # Foundation
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
                    produce a taller, less wide plot. 'square' produces a square plot. 'wide' produces a
                    plot that is much wide than it is tall.
        """
        self.chartProp = chartProp
        self.plotOrientation = plotOrientation
        self.fig = fig
        
        # Dynamically set chart width and height parameters.
        if plotOrientation == 'tall':
            chartWidth = self.chartProp * .7
            chartHeight = self.chartProp * .8
        elif plotOrientation == 'square':
            chartWidth = self.chartProp
            chartHeight = self.chartProp * .8
        elif plotOrientation == 'wide':
            chartWidth = self.chartProp * 1.7
            chartHeight = self.chartProp * .32
        else:            
            chartWidth = self.chartProp
            chartHeight = self.chartProp * .5
        self.fig.set_figheight(chartHeight)
        self.fig.set_figwidth(chartWidth)

    def makeCanvas(self, title = '', xLabel = '', yLabel = '', yShift = 0.8, position = 111):
        """
        Info:
            Description: 
                Create Axes object. Add descriptive attributes such as titles and axis labels,
                set font size and font color. Remove grid. Remove top and right spine. 
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
        
        # Add title.
        ax.set_title(title
                    ,fontsize = 1.999 * self.chartProp if position == 111 else 1.499 * self.chartProp
                    ,color = qpStyle.qpGrey
                    ,loc = 'left'
                    ,pad = 1.667 * self.chartProp)
        ax.tick_params(axis = 'both', colors = qpStyle.qpGrey, labelsize = 1.333 * self.chartProp)

        # Remove grid line and right/top spines.
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add axis labels.
        plt.xlabel(xLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (0.5, 0.5))
        plt.ylabel(yLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (1.0, yShift))
        return ax

    # Standard visualization, matplotlib
    def qp2dScatter(self, x, y, df = None, xUnits = 'f', yUnits = 'f', plotBuffer = True, size = 10
                    , axisLimits = True, color = qpStyle.qpGrey, facecolor = 'w', ax = None):
        """
        Info:
            Description: 
                Create 2-dimensional scatter plot.
            Parameters:
                x : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                y : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                df : Pandas DataFrame, default = None
                    Dataset containing data to be plotted. Can be any size - plotted columns will be 
                    chosen by columns names specified in x, y. 
                xUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                plotBuffer : boolean, default = True
                    Switch for determining whether dynamic plot buffer function is executed.
                size : int or float, default = 10
                    Determines scatter dot size
                axisLimits : boolean, default = True
                    Switch for determining whether dynamic axis limit setting function is executed.
                color : string (color code of some sort), default = qpStyle.qpGrey
                    Determine color of scatter dots
                facecolor : string (color code of some sort), default = 'w'
                    Determine face color of scatter dots
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
        if df is not None:
            x = df[x].values.reshape(-1,1)
            y = df[y].values.reshape(-1,1)
        # Else reshape arrays.
        else:
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
        
        # Plot 2-d scatter.
        plt.scatter(x = x
                    ,y = y
                    ,color = color
                    ,s = size * self.chartProp
                    ,alpha = 0.7
                    ,facecolor = facecolor
                    ,linewidth = 0.167 * self.chartProp
                   )
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)        
    
        # Dynamically set axis lower / upper limits.
        if axisLimits:
            xMin, xMax, yMin, yMax = qpUtil.qpUtilSetAxes(x = x, y = y)        
            plt.axis([xMin, xMax, yMin, yMax])   

        # Create smaller buffer around plot area to prevent cutting off elements.
        if plotBuffer:
            qpUtil.qpUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        # Show figure with tight layout.
        plt.tight_layout()
    
    def qp2dScatterHue(self, x, y, target, label, df = None, xUnits = 'd', yUnits = 'd', plotBuffer = True
                        , size = 10, axisLimits = True, color = qpStyle.qpGrey, facecolor = 'w'
                        , bbox = (1.2, 0.9), ax = None):
        """
        Info:
            Description: 
                Create 2-dimensional scatter plot with a third dimension represented as a color hue in the
                scatter dots.
            Parameters:
                x : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                y : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                target : array or string
                    Either 1-dimensional array of values or a column name in a Pandas DataFrame.
                label : list
                    List of labels describing hue.
                df : Pandas DataFrame, default = None
                    Dataset containing data to be plotted. Can be any size - plotted columns will be 
                    chosen by columns names specified in x, y. 
                xUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                plotBuffer : boolean, default = True
                    Switch for determining whether dynamic plot buffer function is executed.
                size : int or float, default = 10
                    Determines scatter dot size
                axisLimits : boolean, default = True
                    Switch for determining whether dynamic axis limit setting function is executed.
                color : string (color code of some sort), default = qpStyle.qpGrey
                    Determine color of scatter dots
                facecolor : string (color code of some sort), default = 'w'
                    Determine face color of scatter dots
                bbox : tuple of floats, default = (1.2, 0.9)
                    Coordinates for determining legend position
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # If a Pandas DataFrame is passed to function, create x, y and target arrays using columns names 
        # passed into function. Also create X, which is a matrix containing the x, y and target columns.
        if df is not None:
            X = df[[x, y, target]].values
            x = df[x].values
            y = df[y].values
            target = df[target].values
        # Concatenate the x, y and target arrays.
        else:
            X = np.c_[x, y, target]

        # Unique target values.
        targetIds =  np.unique(X[:, 2])
            
        # Loop through sets of target values, labels and colors to create 2-d scatter with hue.
        for targetId, targetLabel, color in zip(targetIds, label, qpStyle.qpColorsHexMid[:len(targetIds)]):
            plt.scatter(x = X[X[:,2] == targetId][:,0]
                        ,y = X[X[:,2] == targetId][:,1]
                        ,color = color
                        ,label = targetLabel
                        ,s = size * self.chartProp
                        ,alpha = 0.7
                        ,facecolor = 'w'
                        ,linewidth = 0.234 * self.chartProp
                    )
        
        # Add legend to figure.
        if label is not None:
            plt.legend(loc = 'upper right'
                       ,bbox_to_anchor = bbox
                       ,ncol = 1
                       ,frameon = True
                       ,fontsize = 1.1 * self.chartProp
                      )
            
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

        # Dynamically set axis lower / upper limits.
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
                Create line plot. Capable of plotting multile lines on the same figure. Also capable of
                adjusting which axis will have the same data for each line and which will have different
                data for each line.
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
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
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
        else:
            x = x.reshape(-1,1) if len(x.shape) == 1 else x
            y = y.reshape(-1,1) if len(y.shape) == 1 else y
        
        # Add line 
        if not yMultiVal:
            for ix in np.arange(x.shape[1]):
                xCol = x[:, ix]
                plt.plot(xCol
                         ,y
                         ,color = linecolor if linecolor is not None else qpStyle.qpColorsHexMid[ix]
                         ,linestyle = linestyle if linestyle is not None else qpStyle.qpLineStyle[0]
                         ,linewidth = 0.247 * self.chartProp
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
                         ,linewidth = 0.247 * self.chartProp
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
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

        # Dynamically set axis lower / upper limits
        if axisLimits:
            xMin, xMax, yMin, yMax = qpUtil.qpUtilSetAxes(x = x, y = y)
            plt.axis([xMin, xMax, yMin, yMax])   
        
        # Create smaller buffer around plot area to prevent cutting off elements.
        if plotBuffer:
            qpUtil.qpUtilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

        # Show figure with tight layout.
        plt.tight_layout()    
    
    def qpBarV(self, x, counts, color = qpStyle.qpColorsHexMid[0], labelRotate = 0, yUnits = 'f', ax = None):
        """
        Info:
            Description: 
                Create vertical bar plot.
            Parameters:
                x : array
                    1-dimensional array of values to be plotted on x-axis representing distinct categories.
                counts : array or string
                    1-dimensional array of value counts for categories.
                color : string (some sort of color code), default = qpStyle.qpColorsHexMid[0]
                    Bar color.
                labelRotate : float or int, default = 0
                    Degrees by which the xtick labels are rotated.
                yUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create vertical bar plot.
        plt.bar(x = x
                ,height = counts
                ,color = color
                ,tick_label = x
                ,alpha = 0.8
            )

        # Rotate x-tick labels.
        plt.xticks(rotation = labelRotate)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, yUnits = yUnits)

        # Resize x-axis labels as needed
        if len(x) > 10 and len(x) <= 20:
            ax.tick_params(axis = 'x', colors = qpStyle.qpGrey, labelsize = 1.2 * self.chartProp)
        elif len(x) > 20:
            ax.tick_params(axis = 'x', colors = qpStyle.qpGrey, labelsize = 0.6 * self.chartProp)
        
    def qpBarH(self, y, counts, color = qpStyle.qpColorsHexMid[0], labelRotate = 45, xUnits = 'f', ax = None):
        """
        Info:
            Description: 
                Create vertical bar plot.
            Parameters:
                y : array
                    1-dimensional array of values to be plotted on x-axis representing distinct categories.
                counts : array or string
                    1-dimensional array of value counts for categories.
                color : string (some sort of color code), default = qpStyle.qpColorsHexMid[0]
                    Bar color.
                labelRotate : float or int, default = 45
                    Degrees by which the xtick labels are rotated.
                xUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within `function.
        """
        # Plot horizontal bar plot.
        plt.barh(y = y
                ,width = counts
                ,color = color
                ,tick_label = y
                ,alpha = 0.8
            )
        
        # Rotate x-tick labels.
        plt.xticks(rotation = labelRotate)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits)

    def qpPairPlot(self, df, vars = None, hue = None, diag_kind = 'auto'):
        """
        Info:
            Description: 
                Create pair plot that produces a grid of scatter plots for all unique pairs of
                continuous features.
            Parameters:
                df : Pandas DataFrame
                    Pandas DataFrame containing data of interest.
                vars : list, default = None
                    List of strings describing columns in Pandas DataFrame to be visualized.
                hue : string, default = None
                    Variable name to be used to introduce third dimension to scatter plots through
                    a color hue.
                diag_kind : string, default = 'auto.
                    Type of plot created along diagonal.
        """
        # Custom plot formatting settings for this particular chart.
        with plt.rc_context({'axes.titlesize' : 3.5 * self.chartProp
                            ,'axes.labelsize' : 1.5 * self.chartProp  # Axis title font size
                            ,'xtick.labelsize' : 1.2 * self.chartProp
                            ,'xtick.major.size' : 0.5 * self.chartProp
                            ,'xtick.major.width' : 0.05 * self.chartProp
                            ,'ytick.labelsize' : 1.2 * self.chartProp
                            ,'ytick.major.size' : 0.5 * self.chartProp
                            ,'ytick.major.width' : 0.05 * self.chartProp
                            ,'figure.facecolor' : qpStyle.qpWhite
                            ,'axes.facecolor': qpStyle.qpWhite
                            ,'axes.spines.left': True
                            ,'axes.spines.bottom': True            
                            ,'axes.edgecolor': qpStyle.qpGrey
                            }):
            
            # Create pair plot.
            g = sns.pairplot(data = df
                            ,vars = vars
                            ,hue = hue 
                            ,diag_kind = diag_kind
                            ,height = 0.2 * self.chartProp
                            ,plot_kws = {'s' : 2.0 * self.chartProp
                                         ,'edgecolor' : None
                                         ,'linewidth' : 1
                                         ,'alpha' : 1.0
                                         ,'marker' : 'o'
                                        }
                            ,diag_kws = {'shade' : True}
                            ,palette = qpStyle.qpColorsHexMid
                            )
            
            # Add custom legend describing hue labels
            if hue is not None:
                
                # Turn off standard legend
                g.fig.legend()
                g.fig.legends = []

                # Add custom legend
                handles = g._legend_data.values()
                labels = g._legend_data.keys()
                g.fig.legend(handles = handles
                            ,labels = labels
                            ,loc = 'upper center'
                            ,markerscale = 0.15 * self.chartProp
                            ,ncol = len(df[hue].unique())
                            ,bbox_to_anchor = (0.5, 1.05, 0, 0)
                            ,prop = {'size' : 1.5 * self.chartProp}
                            )
        
    def qpCorrHeatmap(self, df, annot = True, cols = None, ax = None):
        """
        Info:
            Description:
                Using continuous features, create correlation heatmap. Capable of dropping 
                zeros in select features, where zeros potentially indicate a complete absence
                of the feature.
            Parameters:
                df : Pandas DataFrame
                    Pandas DataFrame
                annot : boolean, default = True
                    Determines whether or not correlation table is annoted wil correlation
                    value or not.
                cols : list, default = None
                    List of strings describing dataframe column. Limits dataframe to select
                    columns.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create correlation matrix
        corrMatrix = df[cols].corr() if cols is not None else df.corr() 
        
        # Create heatmap using correlation matrix
        g = sns.heatmap(corrMatrix
                    ,vmin = -1.0
                    ,vmax = 1.0
                    ,annot = annot
                    ,annot_kws = {'size' : 1.5 * self.chartProp}
                    ,square = True
                    ,ax = ax
                    ,cmap = LinearSegmentedColormap.from_list(name = ''
                                                            ,colors = [qpStyle.qpColorsRgb0Mid[1], 'white', qpStyle.qpColorsRgb0Mid[0]])
                    )

        # Format x and y-tick labels
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp)
        g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp)

        # Customize color bar formatting
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 2.0 * self.chartProp, length = 0)
        cbar.set_ticks([1, -1])

    def qpFacetNum(self, x, color, label, alpha = 0.8):
        """
        Info:
            Description:
                Create histogram of continuous variable. Simple function capable of easy
                iteration through several groupings of a continuous variable that is 
                separated out based on a categorical label. This results in several overlapping
                histograms and can reveal differences in distributions.
            Parameters:
                x : array
                    1-dimensional array of values to be plotted on x-axis.
                color : string (some sort of color code)
                    Determines color of histogram
                label : string
                    Category value label
                alpha : float, default = 0.8
                    Fades histogram bars to create transparent bars            
        """
        # Create histogram.
        plt.hist(x = x
                ,color = color
                ,label = label
                ,alpha = alpha
                )
            
    def qpFacetCat(self, df, feature, labelRotate = 0, yUnits = 'f', xUnits = 's', bbox = (1.2, 0.9), ax = None):       
        """
        Info:
            Description:
                Creates a count plot of a categorical feature and facets that feature by a categorical label.
            Parameters:
                df : Pandas DataFrame
                    Pandas DataFrame
                feature : string
                    String describing column name containing target values
                labelRotate : float or int, default = 45
                    Degrees by which the xtick labels are rotated.
                xUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'd'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                bbox : tuple of floats, default = (1.2, 0.9)
                    Coordinates for determining legend position
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        ixs = np.arange(df.shape[0])
        bar_width = 0.35
        
        featureDict = {}
        for feature in df.columns[1:]:
            featureDict[feature] = df[feature].values.tolist()
        for featureIx, (k, v) in enumerate(featureDict.items()):
            plt.bar(ixs + (bar_width * featureIx)
                    ,featureDict[k]
                    ,bar_width
                    ,alpha = 0.75
                    ,color = qpStyle.qpColorsHexMid[featureIx]
                    ,label = str(k)
                    )
        
        # Custom x-tick labels.
        plt.xticks(ixs[:df.shape[0]] + bar_width / 2, df.iloc[:,0].values)
        
        # Rotate x-tick labels.
        plt.xticks(rotation = labelRotate)
                
        # Add legend to figure.
        plt.legend(loc = 'upper right'
                    ,bbox_to_anchor = bbox
                    ,ncol = 1
                    ,frameon = True
                    ,fontsize = 1.1 * self.chartProp
                    )
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
        
        # Resize x-axis labels as needed.
        if len(featureDict[feature]) > 10 and len(featureDict[feature]) <= 20:
            ax.tick_params(axis = 'x', colors = qpStyle.qpGrey, labelsize = 1.2 * self.chartProp)
            
        elif len(featureDict[feature]) > 20:
            ax.tick_params(axis = 'x', colors = qpStyle.qpGrey, labelsize = 0.6 * self.chartProp)

    def qpProbPlot(self, x, plot):
        """
        Info:
            Description:
                Create plot that visualizes how well a continuous feature's distribution
                conforms to a normal distribution
            Parameters:
                x : array
                    1-dimensional array containing data of a continuous feature
                plot : plot object
                    Plotting object for applying additional formatting.
        """
        stats.probplot(x, plot = plot)
        
        # Override title and axis labels.
        plot.set_title('')
        plt.xlabel('')
        plt.ylabel('')
        
        # Format scatter dots.
        plot.get_lines()[0].set_markerfacecolor(qpStyle.qpWhite)
        plot.get_lines()[0].set_color(qpStyle.qpColorsHexMid[2])
        plot.get_lines()[0].set_markersize(2.0)

        # Format line representing normality.
        plot.get_lines()[1].set_linewidth(2.0)
        plot.get_lines()[1].set_color(qpStyle.qpGrey)
    
    # Standard visualization, seaborn
    def qpDist(self, x, color, yUnits = 'f', xUnits = 'f', fit = None, ax = None):
        """
        Info:
            Description:
                Creates distribution plot for continuous variables, showing counts of a single
                variable. Also overlays a kernel density estimation curve.
            Parameters:
                x : array
                    Data to be plotted.
                color : string (some sort of color code)
                    Determines color of bars, KDE lines.
                xUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                fit : random variabe object, default = None
                    Allows for the addition of another curve. Utilizing 'norm' overlays a normal distribution
                    over the distribution bar chart. Useful for seeing how well, or not, distribution tracks
                    with a normal distrbution.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        g = sns.distplot(a = x
                        ,kde = True
                        ,color = color
                        ,axlabel = False
                        ,fit = fit
                        ,ax = ax)

        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
        
    def qpKde(self, x, color, yUnits = 'f', xUnits = 'f', ax = None):
        """
        Info:
            Description:
                Create kernel density curve for a feature.
            Parameters:
                x : array
                    Data to be plotted.
                color : string (some sort of color code)
                    Determines color of KDE lines.
                xUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                yUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create kernel density estimation line
        g = sns.kdeplot(data = x
                        ,shade = True
                        ,color = color
                        ,legend = None
                        ,ax = ax)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    def qpBoxPlotV(self, x, y, data, color, labelRotate = 0, yUnits = 'f', ax = None):
        """
        Info:
            Description:
                Create vertical box plots. Useful for evaluated a continuous target on the y-axis
                vs. several different category segments on the x-axis
            Parameters:
                x : string
                    Name of independent variable in dataframe. Represents a category
                y : string
                    Name of continuous target variable. 
                data : Pandas DataFrame
                    Pandas DataFrame including both indpedent variable and target variable.
                color : string
                    Determines color of box plot figures. Ideally this object is a color palette,
                    which can be a default seaborn palette, a custom seaborn palette, or a custom
                    matplotlib cmap.
                labelRotate : float or int, default = 45
                    Degrees by which the xtick labels are rotated.
                yUnits : string, default = 'f'
                    Determines units of y-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create vertical box plot.
        g = sns.boxplot(x = x
                        ,y = y
                        ,data = data
                        ,orient = 'v'
                        ,palette = color
                        ,ax = ax).set(
                                    xlabel = None
                                    ,ylabel = None
                                )
        
        # Resize x-axis labels as needed.
        unique = np.unique(data[x])
        if len(unique) > 10 and len(unique) <= 20:
            ax.tick_params(axis = 'x', labelsize = 1.2 * self.chartProp)
        elif len(unique) > 20:
            ax.tick_params(axis = 'x', labelsize = 0.6 * self.chartProp)
        
        # Fade box plot figures by reducing alpha.
        plt.setp(ax.artists, alpha = 0.8)
        
        # Rotate x-tick labels.
        plt.xticks(rotation = labelRotate)

        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, yUnits = yUnits)
            
    def qpBoxPlotH(self, x, y, data, color = qpStyle.qpColorsHexMid, xUnits = 'f', ax = None):
        """
        Info:
            Description:
                Create horizontal box plots. Useful for evaluating a categorical target on the y-axis
                vs. a continuous independent variable on the x-axis
            Parameters:
                x : string
                    Name of independent variable in dataframe. Represents a category
                y : string
                    Name of continuous target variable. 
                data : Pandas DataFrame
                    Pandas DataFrame including both indpedent variable and target variable.
                color : string (some sort of color code), default = qpStyle.qpColorsHexMid
                    Determines color of box plot figures. Ideally this object is a color palette,
                    which can be a default seaborn palette, a custom seaborn palette, or a custom
                    matplotlib cmap.
                xUnits : string, default = 'f'
                    Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create horizontal box plot.
        g = sns.boxplot(x = x
                        ,y = y
                        ,hue = y
                        ,data = data
                        ,orient = 'h'
                        ,palette = color
                        ,ax = ax).set(
                                    xlabel = None
                                    ,ylabel = None
                                )
        
        # Fade box plot figures by reducing alpha.
        plt.setp(ax.artists, alpha = 0.8)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits)
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    
    def qpRegPlot(self, x, y, data, color = qpStyle.qpColorsHexMid[0], x_jitter = None, xUnits = 'f', yUnits = 'f', ax = None):
        """
        Info:
            Description:

            Parameters:
                x : string
                    Name of independent variable in dataframe. Represents a category
                y : string
                    Name of continuous target variable. 
                data : Pandas DataFrame
                    Pandas DataFrame including both indpedent variable and target variable.
                color : string
                    Determines color of dots and regression line.
                labelRotate : float or int, default = 45
                    Degrees by which the xtick labels are rotated.
                yUnits : string, default = 'f'
                    Determines units of y-axis tick labels. 'f' displays float. '%' displays percentages, 
                    '$' displays dollars. 'd' displays real numbers.
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
        """
        # Create vertical box plot.
        g = sns.regplot(x = x
                        ,y = y
                        ,data = data
                        ,x_jitter = x_jitter
                        ,color = color
                        ,scatter_kws = {'alpha' : 0.5}
                        ,ax = ax).set(
                                    xlabel = None
                                    ,ylabel = None
                                )        
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
    
    # Evaluation
    def qpConfusionMatrix(self, yTest, yPred, ax = None):
        """
        Info:
            Description:
                yTest : array
                    1-dimensional array containing true values
                yPred : array
                    1-dimensional array containing predictions
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within `function.
        """
        # Create confution matrix.
        cm = pd.DataFrame(metrics.confusion_matrix(y_true = yTest
                                                    , y_pred = yPred)
                                                    )
        
        # Sort rows and columns in descending order
        cm.sort_index(axis = 1, ascending = False, inplace = True)
        cm.sort_index(axis = 0, ascending = False, inplace = True)
        
        # Apply heatmap styling to confusion matrix.
        sns.heatmap(data = cm
                    ,annot = True
                    ,square = True
                    ,cbar = False
                    ,cmap = 'Blues'
                    ,annot_kws = {'size' : 2.5 * self.chartProp})
        
        # Remove top tick marks and add labels
        ax.xaxis.tick_top()
        plt.xlabel('predicted', size = 40)
        plt.ylabel('actual', size = 40)
        
    def qpRocCurve(self, model, xTrain, yTrain, xTest, yTest, linecolor, ax = None):
        """
        Info:
            Description:
                Plot ROC curve and report AUC
            Parameters:
                model : sklearn model or pipeline
                    model to fit
                xTrain : array
                    Training data to fit model
                yTrain : array
                    Training data to fit model
                xTest : array
                    Test data to return predict_probas
                yTest : array
                    Test data for creating roc_curve
                linecolor : str
                    line color
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within `function.

        """
        # Return prediction probabilities.
        probas = model.fit(xTrain, yTrain).predict_proba(xTest)
        
        # Return false positive rate, true positive rate and thresholds.
        fpr, tpr, thresholds = metrics.roc_curve(y_true = yTest, y_score = probas[:, 1], pos_label = 1)
        
        # Calculate area under the curve using FPR and TPR.
        roc_auc = metrics.auc(fpr, tpr)
        
        # Plot ROC curve.
        self.qpLine(x = fpr
                    ,y = tpr
                    ,label = ['ROC AUC = {:.3f}'.format(roc_auc)]
                    ,linecolor = linecolor
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,bbox = (1.0, 0.8)
                    ,ax = ax
                   )
        
        # Plot 'random guess' line.
        self.qpLine(x = np.array([0, 1])
                    ,y = np.array([0, 1])
                    ,linecolor = qpStyle.qpGrey
                    ,linestyle = '--'
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,ax = ax
                   )
        
        # Plot 'perfection' line.
        self.qpLine(x = np.array([0, 0, 1])
                    ,y = np.array([0, 1, 1])
                    ,linecolor = qpStyle.qpGrey
                    ,linestyle = ':'
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,ax = ax
                   )

    def qpDecisionRegion(self, x, y, classifier, testIdx = None, resolution = 0.001, bbox = (1.2, 0.9), ax = None):
        """
        Info:
            Description:
                Create 2-dimensional chart with shading used to highlight decision regions.
            Parameters:
                X : array
                    m x 2 array containing 2 features.
                y : array
                    m x 1 array containing labels for observations.
                classifier : sklearn model
                    Classifier used to create decision regions.
                testIdx :  tuple, default = None
                    Option parameter for specifying observations to be highlighted
                    as test examples.
                resolution : float, default = 0.001
                    Controls clarity of the graph by setting interval of the arrays 
                    passed into np.meshgrid.
                bbox : tuple of floats, default = (1.2, 0.9)
                    Coordinates for determining legend position
                ax : Axes object, default = None
                    Axes object containing figure elements to be adjusted within function.
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

class MLEDA(QuickPlot):
    """
    Info:
        Description:
            Child class of QuickPlot that contains methods for evaluating and visualizing datasets
            based on the feature type and target variable type.
    """
    def __init__(self, data, removeFeatures = [], overrideCat = None, overrideNum = None, dateFeatures = None, target = None, targetType = None):
        """
        Info:
            Description:

            Parameters:
                data : Pandas DataFrame
                    Input data
                removeFeatures : list, default = []
                    Features to be completely removed from dataset
                overrideCat : list, default = None
                    Preidentified categorical features that would otherwise be labeled as continuous
                overrideCNum : list, default = None
                    Preidentified continuous features that would otherwise be labeled as categorical
                dateFeatures : list, default = None
                    Features comprised of date values, which will need to be handled differently
                target : list, default = None
                    Name of column containing dependent variable
                targetType : list, default = None
                    Target variable type, either 'categorical' or 'continuous
            Attributes:
                X_ : Pandas DataFrame
                    Independent variables
                y_ : Pandas Series
                    Dependent variables
                featuresByDtype_ : dict
                    Dictionary containing two keys, continuous and categorical, each paired with a
                    value that is a list of column names that are of that feature type - continuous or categorical.
        """
        self.data = data
        self.removeFeatures = removeFeatures
        self.overrideCat = overrideCat
        self.overrideNum = overrideNum
        self.dateFeatures = dateFeatures
        self.target = target
        self.targetType = targetType

        # Execute method qpMeasLevel
        if self.target is not None:
            self.X_, self.y_, self.featureByDtype_ = self.qpMeasLevel()
        else:
            self.X_, self.featureByDtype_ = self.qpMeasLevel()

    def qpMeasLevel(self):
        """
        Info:
            Description:
                Isolate independent variables in X_.
                If provided, isolate dependent variable y_.
                Determine level of measurement for each feature as categorical, continuous or date.
        """
        ### Identify target from features
        if self.target is not None:
            self.y_ = self.data[self.target]
            self.X_ = self.data.drop(self.removeFeatures + self.target, axis = 1)
        else:
            self.X_ = self.data.drop(self.removeFeatures, axis = 1)
            
        
        ### Add categorical and continuous keys, and any associated overrides
        self.featureByDtype_ = {}
        
        # Categorical
        if self.overrideCat is None:
            self.featureByDtype_['categorical'] = []
        else:
            self.featureByDtype_['categorical'] = self.overrideCat

            # Change data type to object
            for col in self.overrideCat:
                if self.X_[col].dtype != 'object':
                    self.X_[col] = self.X_[col].apply(str)        
        
        # continuous
        if self.overrideNum is None:
            self.featureByDtype_['continuous'] = []
        else:
            self.featureByDtype_['continuous'] = self.overrideNum
        
        # Date
        if self.dateFeatures is None:
            self.featureByDtype_['date'] = []
        else:
            self.featureByDtype_['date'] = self.dateFeatures
        
        # Combined dictionary values for later filtering
        handled = [i for i in sum(self.featureByDtype_.values(), [])]

        
        ### Categorize remaining columns
        for c in [i for i in self.X_.columns if i not in handled]:
            
            # Identify feature type based on column data type
            if str(self.X_[c].dtype).startswith(('int','float')):
                self.featureByDtype_['continuous'].append(c)
            elif str(self.X_[c].dtype).startswith(('object')):
                self.featureByDtype_['categorical'].append(c)

        ### Return objects
        if self.target is not None:
            return self.X_, self.y_, self.featureByDtype_
        else:
            return self.X_, self.featureByDtype_

    def featureSummary(self):
        """
        Info:
            Description:
                Visualize and summarize dataset features based on feature type (categorical vs. continuous) 
                and target feature type (categorical vs. continuous). Includes data visualizations as well
                as summary tables with descriptive statistics and statistical tests, where applicable.
        """        
        ### Iterate through each feature type and associated feature list
        for k, v in self.featureByDtype_.items():
            
            ### continuous feature summary
            if k == 'continuous':
                print('***********************************************************\n{} columns \n***********************************************************\n'.format(k))
                
                # Iterate through each feature within a feature type
                for feature in v:                
                    
                    # Instantiate charting object
                    p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')

                    ############################################################################################################
                    ### vs. continuous target variable
                    if self.targetType == 'continuous':
                      
                        # Define bivariate dataframe
                        biDf = pd.DataFrame(self.X_[feature]).join(self.y_)
                        
                        # Define summary tables
                        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
                        
                        # If continuous variable has fewer than a set number of unique variables, represent variable
                        # as a categorical variable vs. a continuous target variable
                        if len(np.unique(biDf[feature].values)) <= 20:
                            
                            describeDf = describeDf.rename(columns = {'index' : ''})                        
                            
                            # Bivariate summary statistics
                            biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])                            
                            for featureVal in np.unique(biDf[feature].values):
                                featureSlice = biDf[(biDf[feature] == featureVal) & (biDf[feature].notnull())][feature]
                            
                                biSummStatsDf = biSummStatsDf.append({feature : featureVal
                                                                        ,'Count' : len(featureSlice)
                                                                        ,'Proportion' : len(featureSlice) / len(biDf[feature]) * 100
                                                                        ,'Mean' : np.mean(featureSlice)
                                                                        ,'StdDv' : np.std(featureSlice)
                                                                        }
                                                                    ,ignore_index = True)
                            
                            # Display summary dataframes
                            self.dfSideBySide(dfs = (describeDf, biSummStatsDf), names = ['Univariate stats', 'Bivariate stats'])

                            # Univariate plot
                            uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
                            ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 131)
                            p.qpBarV(x = uniqueVals
                                ,counts = uniqueCounts
                                ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,yUnits = 'f'
                                ,ax = ax)                        

                            # Regression plot
                            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), yShift = 0.8, position = 132)
                            p.qpRegPlot(x = feature
                                        ,y = self.target[0]
                                        ,data = biDf[biDf[feature].notnull()]
                                        ,x_jitter = .1
                                        ,ax = ax)

                            # Bivariate box plot
                            ax = p.makeCanvas(title = 'Box plot - Faceted by\n* {}'.format(feature), yShift = 0.8, position = 133)
                            p.qpBoxPlotV(x = feature
                                        ,y = self.target[0]
                                        ,data = biDf[biDf[feature].notnull()]
                                        ,color = qpStyle.genCmap(len(uniqueVals), [qpStyle.qpColorsHexMid[0], qpStyle.qpColorsHexMid[1], qpStyle.qpColorsHexMid[2]])
                                        ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                        ,ax = ax)

                        # If continuous variable has greater than a set number of unique variables, represent variable
                        # as a continuous variable vs. a continuous target variable
                        else:
                            
                            # Add skew and curtosis to describeDf
                            describeDf = describeDf.append({'index' : 'skew', feature : stats.skew(biDf[feature].values, nan_policy = 'omit')
                                                       }, ignore_index = True)
                            describeDf = describeDf.append({'index' : 'kurtosis', feature : stats.kurtosis(biDf[feature].values, nan_policy = 'omit')
                                                       }, ignore_index = True)
                            describeDf = describeDf.rename(columns = {'index' : ''})
                            
                            # Display summary dataframes
                            display(describeDf)

                            # Univariate plot
                            ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), yShift = 0.8, position = 131)
                            p.qpDist(biDf[(biDf[feature].notnull())][feature].values                                
                                    ,color = qpStyle.qpColorsHexMid[2]
                                    ,yUnits = 'ffff'
                                    ,fit = stats.norm
                                    ,ax = ax)
                        
                            # Probability plot
                            ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), yShift = 0.8, position = 132)
                            p.qpProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                                        ,plot = ax)
                        
                            # Regression plot
                            ax = p.makeCanvas(title = 'Regression plot\n* {}'.format(feature), yShift = 0.8, position = 133)
                            p.qpRegPlot(x = feature
                                        ,y = self.target[0]
                                        ,data = biDf[biDf[feature].notnull()]
                                        ,x_jitter = .1
                                        ,ax = ax)                            
                        plt.show()                        
                    
                    ############################################################################################################
                    ### vs. categorical target variable
                    elif self.targetType == 'categorical':
                        
                        # Bivariate roll-up table
                        biDf = pd.DataFrame(self.X_[feature]).join(self.y_)
                        
                        # Bivariate summary statistics
                        biSummStatsDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion', 'Mean', 'StdDv'])
                        
                        for labl in np.unique(self.y_):
                            featureSlice = biDf[biDf[self.target[0]] == labl][feature]
                        
                            biSummStatsDf = biSummStatsDf.append({feature : labl
                                                                    ,'Count' : len(featureSlice)
                                                                    ,'Proportion' : len(featureSlice) / len(biDf[feature]) * 100
                                                                    ,'Mean' : np.mean(featureSlice)
                                                                    ,'StdDv' : np.std(featureSlice)
                                                                    }
                                                                ,ignore_index = True)
                        
                        # Display summary tables
                        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
                        describeDf = describeDf.append({'index' : 'skew'
                                                        ,feature : stats.skew(biDf[feature].values, nan_policy = 'omit')
                                                       }
                                                ,ignore_index = True)
                        describeDf = describeDf.append({'index' : 'kurtosis'
                                                        ,feature : stats.kurtosis(biDf[feature].values, nan_policy = 'omit')
                                                       }
                                                ,ignore_index = True)
                        describeDf = describeDf.rename(columns = {'index' : ''})

                        # Execute z-test or t-test
                        if len(np.unique(self.y_)) == 2:
                            s1 = biDf[biDf[self.target[0]] == biDf[self.target[0]].unique()[0]][feature]
                            s2 = biDf[biDf[self.target[0]] == biDf[self.target[0]].unique()[1]][feature]
                            if len(s1) > 30 and len(s2) > 30:
                                z, pVal = ztest(s1, s2)
                                
                                statTestDf = pd.DataFrame(data = [{'z-test statistic' : z, 'p-value' : pVal}]
                                                            ,columns = ['z-test statistic', 'p-value']
                                                            ,index = [feature]).round(4)
                            else:
                                t, pVal = stats.ttest_ind(s1, s2)
                                
                                statTestDf = pd.DataFrame(data = [{'t-test statistic' : t, 'p-value' : pVal}]
                                                            ,columns = ['t-test statistic', 'p-value']
                                                            ,index = [feature]).round(4)
                            self.dfSideBySide(dfs = (describeDf, biSummStatsDf, statTestDf)
                                                ,names = ['Univariate stats', 'Bivariate stats', 'Statistical test'])
                        else:
                            self.dfSideBySide(dfs = (describeDf, biSummStatsDf)
                                                ,names = ['Descriptive stats', 'Bivariate stats'])
                            
                        # Univariate plot
                        ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), yShift = 0.8, position = 151)
                        p.qpDist(biDf[(biDf[feature].notnull())][feature].values
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,yUnits = 'ffff'
                                ,ax = ax)
                        
                        # Probability plot
                        ax = p.makeCanvas(title = 'Probability plot\n* {}'.format(feature), yShift = 0.8, position = 152)
                        p.qpProbPlot(x = biDf[(biDf[feature].notnull())][feature].values
                                    ,plot = ax)
                        
                        # Bivariate kernel density plot
                        ax = p.makeCanvas(title = 'KDE - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 153)
                        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target[0]].values)):
                            p.qpKde(biDf[(biDf[feature].notnull()) & (biDf[self.target[0]] == labl)][feature].values
                                    ,color = qpStyle.qpColorsHexMid[ix]
                                    ,yUnits = 'ffff'
                                    ,ax = ax)

                        # Bivariate histogram
                        ax = p.makeCanvas(title = 'Hist - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 154)
                        for ix, labl in enumerate(np.unique(biDf[(biDf[feature].notnull())][self.target[0]].values)):
                            p.qpFacetNum(biDf[(biDf[feature].notnull()) & (biDf[self.target[0]] == labl)][feature].values
                                        ,color = qpStyle.qpColorsHexMid[ix]
                                        ,label = labl
                                        ,alpha = 0.4)

                        # Boxplot histogram
                        ax = p.makeCanvas(title = 'Boxplot - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 155)
                        p.qpBoxPlotH(x = feature
                                    ,y = self.target[0]
                                    ,data = biDf
                                    ,ax = ax)
                        plt.show()

            ############################################################################################################
            ### Categorical feature summary
            elif k == 'categorical':
                print('***********************************************************\n{} columns \n***********************************************************\n'.format(k))
            
                # Iterate through each feature within a feature type
                for feature in v:                
                    
                    # Instantiate charting object
                    p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')

                    ############################################################################################################
                    ### vs. continuous target variable
                    if self.targetType == 'continuous':
                        
                        # Univariate summary
                        uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
                        uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
                        for i, j in zip(uniqueVals, uniqueCounts):
                            uniSummDf = uniSummDf.append({feature : i
                                                    ,'Count' : j
                                                    ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                                    }
                                                ,ignore_index = True)
                        uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
                        
                        # Bivariate summary
                        biDf = pd.DataFrame(self.X_[feature]).join(self.y_)
                        statsDict = {'N' : len, 'Median' : np.nanmedian, 'Mean' : np.nanmean, 'StdDev' : np.nanstd}
                        biSummPivDf = pd.pivot_table(biDf
                                                    ,index = feature
                                                    ,aggfunc = {self.target[0] : statsDict})
                        multiIndex = biSummPivDf.columns
                        singleIndex = pd.Index([i[1] for i in multiIndex.tolist()])
                        biSummPivDf.columns = singleIndex
                        biSummPivDf.reset_index(inplace = True)

                        # Instantiate charting object
                        p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')
                        
                        # Display summary tables
                        self.dfSideBySide(dfs = (uniSummDf, biSummPivDf), names = ['Univariate summary', 'Bivariate summary'])
                        
                        # Univariate plot
                        ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 121)
                        
                        # Select error catching block for resorting labels
                        try:
                            sorted(uniqueVals, key = int)
                        except ValueError:
                            pass
                        else:
                            # Sort uniqueVals/uniqueCounts for bar chart
                            newIx = [sorted(list(uniqueVals), key = int).index(i) for i in list(uniqueVals)]
                            uniqueVals = np.array(sorted(list(uniqueVals), key = int))
                            uniqueCounts = np.array([y for x,y in sorted(zip(newIx, uniqueCounts))])
                        
                            # Sort temporary data frame for box plot
                            biDf[feature] = biDf[feature].astype(int)

                        p.qpBarV(x = uniqueVals
                                ,counts = uniqueCounts
                                ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,yUnits = 'f'
                                ,ax = ax)                 
                                                
                        # Bivariate box plot
                        ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 122)
                        p.qpBoxPlotV(x = feature
                                    ,y = self.target[0]
                                    ,data = biDf[biDf[feature].notnull()].sort_values([feature])
                                    ,color = qpStyle.genCmap(len(uniqueVals), [qpStyle.qpColorsHexMid[0], qpStyle.qpColorsHexMid[1], qpStyle.qpColorsHexMid[2]])
                                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                    ,ax = ax)                        
                        plt.show()
                    
                    ############################################################################################################
                    ### vs. categorical target variable
                    elif self.targetType == 'categorical':

                        # Univariate summary
                        uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
                        uniqueVals, uniqueCounts = np.unique(self.X_[self.X_[feature].notnull()][feature], return_counts = True)
                        for i, j in zip(uniqueVals, uniqueCounts):
                            uniSummDf = uniSummDf.append({feature : i
                                                    ,'Count' : j
                                                    ,'Proportion' : j / np.sum(uniqueCounts) * 100
                                                    }
                                                ,ignore_index = True)
                        uniSummDf = uniSummDf.sort_values(by = ['Proportion'], ascending = False)
                        
                        # Bivariate summary
                        biDf = pd.DataFrame(self.X_[feature]).join(self.y_)
                        biSummDf = pd.DataFrame(self.X_[feature]).join(self.y_)\
                                            .groupby([feature] + self.y_.columns.tolist()).size().reset_index()\
                                            .pivot(columns = self.y_.columns.tolist()[0], index = feature, values = 0)
                        multiIndex = biSummDf.columns
                        singleIndex = pd.Index([i for i in multiIndex.tolist()])
                        biSummDf.columns = singleIndex
                        biSummDf.reset_index(inplace = True)

                        # Instantiate charting object
                        p = QuickPlot(fig = plt.figure(), chartProp = 15, plotOrientation = 'wide')
                        
                        # Display summary tables
                        self.dfSideBySide(dfs = (uniSummDf, biSummDf), names = ['Univariate summary', 'Biivariate summary'])
                        
                        # Univariate plot
                        ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 121)
                        
                        p.qpBarV(x = uniqueVals
                                ,counts = uniqueCounts
                                ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,yUnits = 'f'
                                ,ax = ax)                        
                        
                        # Bivariate plot
                        ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 122)
                        p.qpFacetCat(df = biSummDf
                                    ,feature = feature
                                    ,labelRotate = 90 if len(uniqueVals) >= 4 else 0
                                    ,ax = ax)
                        plt.show()

    def dateTransformer(self):
        """
        Info:
            Description:

            Parameters:

        """
        pass
        # day of week
        # hour of day
        # morning/afternoon/evening/graveyard
        # weekend vs weekday
    
    def numFeatureFormatter(self):
        """
        Info:
            Description:

            Parameters:
        """
        pass

    def catFeatureFormatter(self):
        """
        Info:
            Description:

            Parameters:
        """
        pass

    def dfSideBySide(self, dfs, names = []):
        """
        Info:
            Description:

            Parameters:
        
        """
        html_str = ''
        if names:
            html_str += ('<tr>' + 
                        ''.join(f'<td style="text-align:center">{name}</td>' for name in names) + 
                        '</tr>')
        html_str += ('<tr>' + 
                    ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' 
                            for df in dfs) + 
                    '</tr>')
        html_str = f'<table>{html_str}</table>'
        html_str = html_str.replace('table','table style="display:inline"')
        display_html(html_str, raw=True)

    def residualPlot(self):
        # what other functions to evaluate performance?
        pass