import seaborn as sns

vizColorsHexLight = ['#8BA4D9', '#D8F894', '#FFDC99', '#E78BC1', '#BC86D9', '#FFC799', '#7DD1D1']
vizColorsHexMidLight = ['#466DC1', '#BCF347', '#FFC24B', '#D84099', '#9340C1', '#FF9C4B', '#35B4B4']
vizColorsHexMid = ['#1449BB', '#A4F201', '#FFA901', '#D5017D', '#7C0BBB', '#FF7401', '#01ADAD']
vizColorsHexMidDark = ['#0A3389', '#82C000', '#CD8700', '#A3005F', '#5A0589', '#CD5D00', '#007B7B']
vizColorsHexDark = ['#04215D', '#588200', '#8B5C00', '#6E0041', '#3D025D', '#8B3F00', '#005353']

vizColorsRgbLight = [(139,164,217), (216,248,148), (255,220,153), (231,139,193), (188,134,217), (255,199,153), (125,209,209)]
vizColorsRgbMidLight = [( 70,109,193), (188,243, 71), (255,194, 75), (216, 64,153), (147, 64,193), (255,156, 75), ( 53,180,180)]
vizColorsRgbMid = [( 20, 73,187), (164,242,  1), (255,169,  1), (213,  1,125), (124, 11,187), (255,116,  1), (  1,173,173)]
vizColorsRgbMidDark = [( 10, 51,137), (130,192,  0), (205,135,  0), (163,  0, 95), ( 90,  5,137), (205, 93,  0), (  0,123,123)]
vizColorsRgbDark = [(  4, 33, 93), ( 88,130,  0), (139, 92,  0), (110,  0, 65), ( 61,  2, 93), (139, 63,  0), (  0, 83, 83)]

vizColorsRgb0Light = [(0.545,0.643,0.851), (0.847,0.973,0.58), (1,0.863,0.6), (0.906,0.545,0.757), (0.737,0.525,0.851), (1,0.78,0.6), (0.49,0.82,0.82)]
vizColorsRgb0MidLight = [(0.275,0.427,0.757), (0.737,0.953,0.278), (1,0.761,0.294), (0.847,0.251,0.6), (0.576,0.251,0.757), (1,0.612,0.294), (0.208,0.706,0.706)]
vizColorsRgb0Mid = [(0.078,0.286,0.733), (0.643,0.949,0.004), (1,0.663,0.004), (0.835,0.004,0.49), (0.486,0.043,0.733), (1,0.455,0.004), (0.004,0.678,0.678)]
vizColorsRgb0MidDark = [(0.039,0.2,0.537), (0.51,0.753,0), (0.804,0.529,0), (0.639,0,0.373), (0.353,0.02,0.537), (0.804,0.365,0), (0,0.482,0.482)]
vizColorsRgb0Dark = [(0.016,0.129,0.365), (0.345,0.51,0), (0.545,0.361,0), (0.431,0,0.255), (0.239,0.008,0.365), (0.545,0.247,0), (0,0.325,0.325)]

vizLineStyle = ['-','--','-.',':','-','--','-.',':','-','--','-.',':'] 

vizMarkers = ('s','o','v','x','^')
        
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