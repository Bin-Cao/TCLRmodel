if __name__ == "__main__":

    import sys
    # Configure a temporary address
    # locdir is the address of your computer where the folder 'TreeClassifierforLinearRegression' is located
    locdir = 'D:\\TCLRUPLOAD\\Source code'
    sys.path.append(locdir)
    import TCLRmodel.TCLRalgorithm as TCLR



    dataSet = "testdata.csv"
    # :param dataSet：the input dataset
    correlation = 'PearsonR(+)'
    #  :param correlation : {'PearsonR(+)','PearsonR(-)',''MIC','R2'}，default PearsonR(+)
    #     * PearsonR: (+)(-). for linear relationship
    #     * MIC for no-linear relationship
    #     * R2 for no-linear relationship、
    minsize = 3
    #  :param minsize : a int number, the minimum number of data in each leave.
    threshold = 0.9
    #  :param threshold : a float, less than or equal to 1, default 0.95 for PearsonR.
    #   To avoid overfitting, threshold = 0.5 is suggested for MIC 0.5.
    mininc = 0.01
    #  :param mininc :Minimum expected gain of objective function



    TCLR.start(dataSet, correlation, minsize, threshold, mininc)



