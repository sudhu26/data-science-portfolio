import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sklearn.metrics as metrics
import sklearn.preprocessing as prepocessing

import quickplot.qpStyle as qpStyle
import quickplot.qpUtil as qpUtil

from IPython.display import display_html
    
sns.set_style('whitegrid')

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
        ax.set_title(title
                    ,fontsize = 1.999 * self.chartProp if position == 111 else 1.499 * self.chartProp
                    ,color = qpStyle.qpGrey
                    ,loc = 'left'
                    ,pad = 1.667 * self.chartProp)
        ax.tick_params(axis = 'both', colors = qpStyle.qpGrey, labelsize = 1.333 * self.chartProp)

        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add axis labels
        plt.xlabel(xLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (0.5, 0.5))
        plt.ylabel(yLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (1.0, yShift))
        return ax

    def qp2dScatter(self, x, y, df = None, xUnits = 'f', yUnits = 'f', plotBuffer = True
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
            x = df[x].values.reshape(-1,1)
            y = df[y].values.reshape(-1,1)
        else:
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
        
        # 2d scatter
        plt.scatter(x = x
                    ,y = y
                    ,color = color
                    ,s = 10 * self.chartProp
                    ,alpha = 0.7
                    ,facecolor = facecolor
                    ,linewidth = 0.167 * self.chartProp
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
    
    def qpBar(self, x, counts, labelRotate = 0, log = False, yUnits = 'f', xUnits = 'f', ax = None):
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
                ,height = counts
                ,color = qpStyle.qpColorsHexMid[0]
                ,tick_label = x
                ,alpha = 0.8
            )

        plt.xticks(rotation = labelRotate)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
        
    def qpBarH(self, y, counts, color = qpStyle.qpColorsHexMid[0], labelRotate = 45, log = False
                , yUnits = 'f', xUnits = 'f', ax = None):
        """
        Info:
            Description:

            Parameters:
                x : Array
                y : Array
                xLabels : List
                log : boolean
        """
        plt.barh(y = y
                ,width = counts
                ,color = color
                ,tick_label = y
                ,alpha = 0.8
            )
        plt.xticks(rotation = labelRotate)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    def qpConfusionMatrix(self, yTest, yPred, ax = None):
        """
        Info:
            Description:

            Parameters:

        """
        cm = pd.DataFrame(metrics.confusion_matrix(y_true = yTest
                                                    , y_pred = yPred)
                                                    )
        cm.sort_index(axis = 1, ascending = False, inplace=True)
        cm.sort_index(axis = 0, ascending = False, inplace=True)
        sns.heatmap(data = cm
                    ,annot = True
                    ,square = True
                    ,cbar = False
                    ,cmap = 'Blues'
                    ,annot_kws = {"size": 2.5 * self.chartProp})
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

        """
        probas = model.fit(xTrain, yTrain).predict_proba(xTest)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = yTest, y_score = probas[:, 1], pos_label = 1)
        roc_auc = metrics.auc(fpr, tpr)
        self.qpLine(x = fpr
                    ,y = tpr
                    ,label = ['ROC AUC = {:.3f}'.format(roc_auc)]
                    ,linecolor = linecolor
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,bbox = (1.0, 0.8)
                    ,ax = ax
                   )
        self.qpLine(x = np.array([0, 1])
                    ,y = np.array([0, 1])
                    ,linecolor = qpStyle.qpGrey
                    ,linestyle = '--'
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,ax = ax
                   )
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

    def qpPairPlot(self, df, vars = None, hue = None, diag_kind = 'auto'):
        """
        Info:
            Description: 
                a        
            Parameters:
                a : a
                    a
                a : a
                    a
        """
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

            Parameters:

        """
        corrMatrix = df[cols].corr() if cols is not None else df.corr() 
        
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

        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp)
        g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp)

        # Customize color bar formatting
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 2.0 * self.chartProp, length = 0)
        cbar.set_ticks([1, -1])

    def qpDist(self, x, color, yUnits = 'f', xUnits = 'f', ax = None):
        """

        """
        g = sns.distplot(a = x
                        ,kde = True
                        ,color = color
                        ,axlabel = False
                        ,ax = ax)

        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
        
    def qpKde(self, x, color, yUnits = 'f', xUnits = 'f', ax = None):
        """

        """
        g = sns.kdeplot(data = x
                        ,shade = True
                        ,color = color
                        ,legend = None
                        ,ax = ax)

        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    def qpBoxPlot(self, x, y, data, color = qpStyle.qpColorsHexMid, xUnits = 'f', ax = None):
        """

        """
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
        plt.setp(ax.artists, alpha = 0.8)
        
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits)
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)

    def qpFacetNum(self, x, color, label, alpha = 0.8):
        """
        Info:
            Description:

            Parameters:
                x : Array
                y : Array
                xLabels : List
                log : boolean, default = True
        """
        plt.hist(x = x
                ,color = color
                ,label = label
                ,alpha = alpha
                )
            
    def qpFacetCat(self, df, feature, yUnits = 'f', xUnits = 's', bbox = (1.2, 0.9), ax = None):       

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
        
        # Custom X-tick labels
        plt.xticks(ixs[:df.shape[0]] + bar_width / 2, df.iloc[:,0].values)
        
        # Add legend to figure
        plt.legend(loc = 'upper right'
                    ,bbox_to_anchor = bbox
                    ,ncol = 1
                    ,frameon = True
                    ,fontsize = 1.1 * self.chartProp
                    )
        # Axis tick label formatting.
        qpUtil.qpUtilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
        
class MLEDA(QuickPlot):


    def __init__(self, data, removeFeatures = [], skipFeatures = [], dateFeatures = [], target = None):
        """
        Info:
            Description:

            Parameters:
                data : Pandas DataFrame
                    Input data
                removeFeatures : list, default = []
                    Features to be completely removed from dataset
                skipFeatures : list, default = []
                    Features to be skipped
                dateFeatures : list, default = []
                    Features comprised of date values
                target : list, default = []
                    Name of column containing dependent variable
            Attributes:
                X_ : Pandas DataFrame

                y_ : Pandas Series

                featuresByDtype_ : dict

        """
        self.data = data
        self.removeFeatures = removeFeatures
        self.skipFeatures = skipFeatures
        self.dateFeatures = dateFeatures
        self.target = target

        if self.target is not None:
            self.X_, self.y_, self.featureByDtype_ = self.qpMeasLevel()
        else:
            self.X_, self.featureByDtype_ = self.qpMeasLevel()

    def qpMeasLevel(self):
        """
        Info:
            Description:
                Capture independent variable in X.
                If provided, capture dependent variable y.
                Determine level of measurement for each feature in X.
        """
        # Identify target from features
        if self.target is not None:
            self.y_ = self.data[self.target]
            self.X_ = self.data.drop(self.removeFeatures + self.target, axis = 1)
        else:
            self.X_ = self.data.drop(self.removeFeatures, axis = 1)
            
        self.featureByDtype_ = {}
        for c in [c for c in self.X_.columns if c not in self.skipFeatures]:
            # Identify feature type
            if len(self.X_[c].unique()) == 2:
                dataType = 'bool'
            elif c in self.dateFeatures:
                dataType = 'date'
            else:
                dataType = str(self.X_[c].dtype)

            # Add feature to dictionary
            if dataType not in self.featureByDtype_.keys():
                self.featureByDtype_[dataType] = [c]
            else:
                self.featureByDtype_[dataType].append(c)

        if self.target is not None:
            return self.X_, self.y_, self.featureByDtype_
        else:
            return self.X_, self.featureByDtype_

    def featureSummary(self, skipDtype = ['date']):
        """

        """        
        # Iterate through each feature type and associated feature list
        for k, v in self.featureByDtype_.items():
            if k not in skipDtype:
                print('**********\n{} columns'.format(k))
                
                # Iterate through each feature within a feature type
                for feature in v:                
                    
                    # Univariate summary
                    uniSummDf = pd.DataFrame(columns = [feature, 'Count', 'Proportion'])
                    unique, unique_counts = np.unique(self.X_[feature], return_counts = True)
                    for i, j in zip(unique, unique_counts):
                        uniSummDf = uniSummDf.append({feature : i
                                                ,'Count' : j
                                                ,'Proportion' : j / np.sum(unique_counts) * 100
                                                }
                                            ,ignore_index = True)
                    
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
                    
                    # Boolean feautes
                    if k in ['bool']:
                        
                        # Display summary tables
                        self.dfSideBySide(dfs = (uniSummDf, biSummDf), names = ['df 1', 'df2'])
                        
                        # Univariate plot
                        ax = p.makeCanvas(title = 'Univariate\n* {}'.format(feature), yShift = 0.8, position = 121)
                        p.qpBarH(y = unique
                                ,counts = unique_counts
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,labelRotate = 45
                                ,xUnits = 'f'
                                ,yUnits = 's'
                                ,ax = ax
                                )
                        
                        # Bivariate plot
                        ax = p.makeCanvas(title = 'Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 122)
                        p.qpFacetCat(df = biSummDf
                                    ,feature = feature
                                    ,ax = ax)

                    # Numeric features
                    elif k in ['int32','int64','float32','float64']:
                        
                        # Display summary tables
                        describeDf = pd.DataFrame(biDf[feature].describe()).reset_index()
                        self.dfSideBySide(dfs = (describeDf, describeDf), names = ['df1', 'df1'])

                        # Univariate plot
                        ax = p.makeCanvas(title = 'Dist/KDE - Univariate\n* {}'.format(feature), yShift = 0.8, position = 141)
                        p.qpDist(self.X_[feature]
                                ,color = qpStyle.qpColorsHexMid[2]
                                ,yUnits = 'ffff'
                                ,ax = ax)
                        
                        # Bivariate kernel density plot
                        ax = p.makeCanvas(title = 'KDE - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 142)
                        for ix, labl in enumerate(np.unique(self.y_)):
                            p.qpKde(biDf[biDf[self.target[0]] == labl][feature]
                                    ,color = qpStyle.qpColorsHexMid[ix]
                                    ,yUnits = 'ffff'
                                    ,ax = ax)
                        
                        # Bivariate histogram
                        ax = p.makeCanvas(title = 'Hist - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 143)
                        for ix, labl in enumerate(np.unique(self.y_)):
                            p.qpFacetNum(biDf[biDf[self.target[0]] == labl][feature]
                                        ,color = qpStyle.qpColorsHexMid[ix]
                                        ,label = labl
                                        ,alpha = 0.4)

                        # Boxplot histogram
                        ax = p.makeCanvas(title = 'Boxplot - Faceted by target\n* {}'.format(feature), yShift = 0.8, position = 144)
                        p.qpBoxPlot(x = feature
                                    ,y = self.target[0]
                                    ,data = biDf
                                    ,ax = ax)

                    # String / object features
                    elif k == 'object':
                        p.qpBar(x = unique
                                ,counts = unique_counts
                                ,labelRotate = 45
                                ,xUnits = 's'
                                ,yUnits = 'f'
                                ,ax = ax
                                )
                    plt.show()

    def featureFormatter(self):
        pass

    def dfSideBySide(self, dfs, names = []):
        # html_str = ''
        # for df in args:
        #     html_str += df.to_html()
        # display_html(html_str.replace('table','table style="display:inline"'), raw = True)
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