import seaborn as sns

vizColors = ['#FF4F00' # orange
            ,'#7F5EBA' # purple
            ,'#FFC02E' # yellow
            ,'#9BD020' # green
            ,'#01ADAD'] # teal

vizLineStyle = ['-'
            ,'--'
            ,'-.'
            ,':'
            ,'-'
            ,'--'
            ,'-.'
            ,':'
            ,'-'
            ,'--'
            ,'-.'
            ,':'] 

vizWhite = (255 / 255, 255 / 255, 255 / 255)
vizGrey = (105 / 255, 105 / 255, 105 / 255)

# rc parameters
rcGrey = {'axes.titlesize' : 50.0
            ,'axes.labelsize' : 40.0   # Axis title font size
            ,'axes.facecolor': vizWhite
            ,'axes.edgecolor': vizWhite
            ,'axes.grid': False
            ,'axes.axisbelow': True
            ,'axes.labelcolor': vizGrey
            ,'axes.spines.left': True
            ,'axes.spines.bottom': True
            ,'axes.spines.right': False
            ,'axes.spines.top': False
                
            ,'xtick.labelsize' : 25.0
            ,'xtick.color': vizGrey
            ,'xtick.direction': 'out'
            ,'xtick.bottom': True
            ,'xtick.top': False
            ,'xtick.major.size' : 10.0
            ,'xtick.major.width' : 3.0
                
            ,'ytick.labelsize' : 25.0
            ,'ytick.color': vizGrey
            ,'ytick.direction': 'out'
            ,'ytick.left': True
            ,'ytick.right': False
            ,'ytick.major.size' : 10.0
            ,'ytick.major.width' : 3.0
                
            ,'font.family': ['Arial']
            }