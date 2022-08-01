print(' Tree classifier for linear regression \n 8 August 2021, version 1, Bin Cao, MGI, SHU, Shanghai, CHINA.')

__version__ = '1.4.11'
__description__ = 'Tree-Classifier for linear regression'
__author__ = 'Bin Cao'
__author_email__ = 'baco@shu.edu.cn'
__url__ = 'https://github.com/Bin-Cao/TCLRmodel'


"""
The entire feature space is divided into disjointed unit intervals by hyperplanes parallel to the coordinate axes.
In each partition, TCLR models target y as the function of a feature
TCLR choses the features and split-point to attain the best fit and recursive binary partitions the space,
until some stopping rules are applied.


Algorithm Patent No. : 2021SR1951267, China
Reference : Domain knowledge guided interpretive machine learning ——  Formula discovery for the oxidation behavior of Ferritic-Martensitic steels in supercritical water. Bin Cao et al., 2022, JMI, journal paper.
DOI : 10.20517/jmi.2022.04

"""