from TCLR import TCLRalgorithm as model
"""
# :param dataSet：the input dataset
# :param correlation : {'PearsonR(+)','PearsonR(-)',''MIC','R2'}，default PearsonR(+)
#     * PearsonR: (+)(-). for linear relationship
#     * MIC for no-linear relationship
#     * R2 for no-linear relationship
#  :param minsize : a int number, minimum unique values for linear features of data on each leaf.
#  :param threshold : a float, less than or equal to 1, default 0.95 for PearsonR.
#                     to avoid overfitting, threshold = 0.5 is suggested for MIC.
#  :param mininc :Minimum expected gain of objective function
"""
dataSet = "testdata.csv"
correlation = 'PearsonR(+)'
minsize = 3
threshold = 0.9
mininc = 0.01
model.start(dataSet, correlation, minsize, threshold, mininc)



