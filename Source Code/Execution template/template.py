#coding=utf-8
from TCLR import TCLRalgorithm as model
"""
    :param correlation : {'PearsonR(+)','PearsonR(-)',''MIC','R2'}，default PearsonR(+).
            Methods:
            * PearsonR: (+)(-). for linear relationship.
            * MIC for no-linear relationship.
            * R2 for no-linear relationship.

    :param minsize : a int number (default=3), minimum unique values for linear features of data on each leaf.
    
    :param threshold : a float (default=0.9), less than or equal to 1, default 0.95 for PearsonR.
            In the process of dividing the dataset, the smallest relevant index allowed in the you research.
            To avoid overfitting, threshold = 0.5 is suggested for MIC 0.5.
    
    :param mininc : Minimum expected gain of objective function (default=0.01)

    :param gplearn : Whether to call the embeded gplearn package of TCLR to regress formula (default=False).
    
    :param population_size : integer, optional (default=500), the number of programs in each generation.
    
    :param generations : integer, optional (default=100),the number of generations to evolve.

    :param verbose : int, optional (default=0). Controls the verbosity of the evolution building process.
    
    :param metric : str, optional (default='mean absolute error')
            The name of the raw fitness metric. Available options include:
            - 'mean absolute error'.
            - 'mse' for mean squared error.
            - 'rmse' for root mean squared error.
            - 'pearson', for Pearson's product-moment correlation coefficient.
            - 'spearman' for Spearman's rank-order correlation coefficient.
    
    :param function_set : iterable, optional (default=['add', 'sub', 'mul', 'div', 'log', 'sqrt', 
                                               'abs', 'neg','inv','sin','cos','tan', 'max', 'min'])
            The functions to use when building and evolving programs. This iterable can include strings 
            to indicate either individual functions as outlined below.
            Available individual functions are:
            - 'add' : addition, arity=2.
            - 'sub' : subtraction, arity=2.
            - 'mul' : multiplication, arity=2.
            - 'div' : protected division where a denominator near-zero returns 1.,
                arity=2.
            - 'sqrt' : protected square root where the absolute value of the
                argument is used, arity=1.
            - 'log' : protected log where the absolute value of the argument is
                used and a near-zero argument returns 0., arity=1.
            - 'abs' : absolute value, arity=1.
            - 'neg' : negative, arity=1.
            - 'inv' : protected inverse where a near-zero argument returns 0.,
                arity=1.
            - 'max' : maximum, arity=2.
            - 'min' : minimum, arity=2.
            - 'sin' : sine (radians), arity=1.
            - 'cos' : cosine (radians), arity=1.
            - 'tan' : tangent (radians), arity=1.

    Algorithm Patent No. : 2021SR1951267, China
    Reference : Domain knowledge guided interpretive machine learning ——  Formula discovery for the oxidation behavior of Ferritic-Martensitic steels in supercritical water. Bin Cao et al., 2022, JMI, journal paper.
    DOI : 10.20517/jmi.2022.04
"""

dataSet = "testdata.csv"
correlation = 'PearsonR(+)'
minsize = 3
threshold = 0.9
mininc = 0.01
gplearn = True
population_size = 500
generations = 100
verbose = 1 
metric = 'mean absolute error'
function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg','inv','sin','cos','tan', 'max', 'min']


model.start(dataSet, correlation, minsize, threshold, mininc, 
            gplearn, population_size,generations,verbose,metric,function_set)



