import seaborn as sns

# custom colors
colWhite = (255 / 255, 255 / 255, 255 / 255)
colTitleGrey = (105 / 255, 105 / 255, 105 / 255)

# rc parameters
rcGrey = {'axes.titlesize' : 50.0
     ,'axes.labelsize' : 40.0   # Axis title font size
     ,'axes.facecolor': colWhite
     ,'axes.edgecolor': colWhite
     ,'axes.grid': False
     ,'axes.axisbelow': True
     ,'axes.labelcolor': colTitleGrey
     ,'axes.spines.left': True
     ,'axes.spines.bottom': True
     ,'axes.spines.right': False
     ,'axes.spines.top': False
           
     ,'xtick.labelsize' : 25.0
     ,'xtick.color': colTitleGrey
     ,'xtick.direction': 'out'
     ,'xtick.bottom': True
     ,'xtick.top': False
     ,'xtick.major.size' : 10.0
     ,'xtick.major.width' : 3.0
           
     ,'ytick.labelsize' : 25.0
     ,'ytick.color': colTitleGrey
     ,'ytick.direction': 'out'
     ,'ytick.left': True
     ,'ytick.right': False
     ,'ytick.major.size' : 10.0
     ,'ytick.major.width' : 3.0
           
     ,'font.family': ['Arial']
     }