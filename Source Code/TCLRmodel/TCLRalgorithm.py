"""
    Tree Classifier for Linear Regression (TCLR) V1.2.0

    TCLR is a novel tree model proposed by Prof.T-Y Zhang and Mr.Bin Cao et al. to capture the functional relationships
    between features and target, which partitions the feature space into a set of rectangles, and embody a specific function in each one.
    It is conceptually simple yet powerful for distinguishing mechanisms. The entire feature space is divided into disjointed
    unit intervals by hyperplanes parallel to the coordinate axes. In each partition, we model target y as the function
    of a feature xj (j = 1,⋯,m), linear function used in our studied problem.

    It is worth noting that TCLR has the function of data screening by discarding which cannot be modeled functionally
    on some resulting leaves. We chose the features and split-point to attain the best fit and recursive binary partitions the space,
    until some stopping rules are applied.

    Patent No. : 2021SR1951267, China
    Reference : Domain knowledge guided interpretive machine learning ——  Formula discovery for the oxidation behavior of Ferritic-Martensitic steels in supercritical water. Bin Cao et al., 2022, JMI, journal paper.
    DOI : 10.20517/jmi.2022.04
"""

import math
import numpy as np
import pandas as pd
from graphviz import Digraph
from scipy import stats
from minepy import MINE
import time
import os



# Define the basic structure of a Tree Model - Node
class Node:
    def __init__(self, data):
        self.data = data
        self.lc = None
        self.rc = None
        self.slope = None
        self.size = data.shape[0]
        self.R = 0
        self.bestFeature = 0
        self.bestValue = 0
        self.leaf_no = -1
        os.makedirs('Segmented', exist_ok=True)


"""
    The evaluation factor for capture the functional relationship between feature and response
    1>
    PearsonR:
    Pearson correlation coefficient, also known as Pearson's r, the Pearson product-moment correlation coefficient.
    PearsonR is a measure of linear correlation between two sets of data. 
    PearsonR = Cov(X,Y) / (sigmaX * sigmaY)
   
    2>
    MIC:
    The maximal information coefficient (MIC). MIC captures a wide range of associations both functional and not, 
    and for functional relationships provides a score that roughly equals the coefficient of determination (R2) of 
    the data relative to the regression function.  MIC belongs to a larger class of maximal information-based 
    nonparametric exploration (MINE) statistics for identifying and classifying relationship.  
    Reference : Reshef, D. N., Reshef, Y. A., Finucane, H. K., Grossman, S. R., McVean, G., Turnbaugh, P. J., ... 
    and Sabeti, P. C. (2011). Detecting novel associations in large data sets. science, 334(6062), 1518-1524.
    
    3>
    R2:
    In statistics, the coefficient of determination, denoted R2 or r2 and pronounced "R squared", 
    is the proportion of the variation in the dependent variable that is predictable from the independent variable(s).
    t is a statistic used in the context of statistical models whose main purpose is either the prediction of future 
    outcomes or the testing of hypotheses, on the basis of other related information. It provides a measure of how well
    observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.
    Definition from Wikipedia ：https://en.wikipedia.org/wiki/Coefficient_of_determination
    R2 = 1 - SSres / SStot. Its value may be a negative one for poor correlation.
 
"""


def PearsonR(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    if len(X) > 1:
        for i in range(0, len(X)):
            diffXXBar = X[i] - xBar
            diffYYBar = Y[i] - yBar
            SSR += (diffXXBar * diffYYBar)
            varX += diffXXBar ** 2
            varY += diffYYBar ** 2
        SST = math.sqrt(varX * varY)
    else:
        SST = 1
        SSR = 0
    if SST == 0:
        return 0
    return SSR / SST


def MIC(X, Y):
    if len(X) > 0:
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(X, Y)
        return mine.mic()
    else:
        MICs = 0
        return MICs


def R2(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    if len(X) > 0:
        a = (X - np.mean(Y)) ** 2
        SStot = np.sum(a)
        b = (X - Y) ** 2
        SSres = np.sum(b)
        r2 = 1 - SSres / SStot
        return r2
    else:
        r2 = -10
        return r2


# Split the DataSet in a specific node
def splitDataSet(dataSet, axis, value):
    retDataSetA = []
    retDataSetB = []
    for featVec in dataSet:
        if featVec[axis] <= value:
            retDataSetA.append(featVec)
        else:
            retDataSetB.append(featVec)
    return np.array(retDataSetA), np.array(retDataSetB)


# Capture the functional relationships between features and target
# Partitions the feature space into a set of rectangles,
def createTree(dataSet, feats, leaf_no, correlation='PearsonR(+)', minsize=3, threshold=0.95, mininc=0.01):
    """

    The entire feature space is divided into disjointed unit intervals by hyperplanes parallel to the coordinate axes.
    In each partition, we model target y as the function of a feature
    We chose the features and split-point to attain the best fit and recursive binary partitions the space,
    until some stopping rules are applied

    ----------
    :param dataSet：the input dataset
    :param feats: feature names from dataset
    :param leaf_no: serial number of leaves
    :param correlation : {'PearsonR(+)','PearsonR(-)',''MIC','R2'}，default PearsonR(+)
        Methods:
        * PearsonR: (+)(-). for linear relationship
        * MIC for no-linear relationship
        * R2 for no-linear relationship
    :param minsize : a int number, the minimum number of data in each leave.
    :param threshold : a float, less than or equal to 1, default 0.95 for PearsonR
        In the process of dividing the dataset, the smallest relevant index allowed in the you research.
        To avoid overfitting, threshold = 0.5 is suggested for MIC 0.5.
    :param mininc :Minimum expected gain of objective function
    ----------

    """
    # It is a  positive linear relationship
    if correlation == 'PearsonR(+)':
        node = Node(dataSet)
        # Initial R0
        bestR = PearsonR(dataSet[:, -2], dataSet[:, -1])
        node.R = bestR
        __slope = stats.linregress(dataSet[:, -2], dataSet[:, -1])[0]
        node.slope = __slope
        if bestR >= threshold:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)
            return node, leaf_no
        # Leave the last two columns of DataSet, a feature of interest and a response
        numFeatures = len(dataSet[0]) - 2
        splitSuccess = False
        bestFeature = -1
        bestValue = 0

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = sorted(list(set(featList)))

            for value in range(len(uniqueVals) - 1):
                subDataSetA, subDataSetB = splitDataSet(dataSet, i, uniqueVals[value])

                if subDataSetA.shape[0] <= minsize - 1 or subDataSetB.shape[0] <= minsize - 1:
                    continue

                newRa = PearsonR(subDataSetA[:, -2], subDataSetA[:, -1])
                newRb = PearsonR(subDataSetB[:, -2], subDataSetB[:, -1])

                R = (newRa + newRb) / 2

                if R - bestR >= mininc:
                    splitSuccess = True
                    bestR = R
                    lc = subDataSetA
                    rc = subDataSetB
                    bestFeature = i
                    bestValue = uniqueVals[value]

        # The recursive boundary is unable to find a division node that can increase factor(R, MIC, R2) by mininc or more.
        if splitSuccess:
            node.lc, leaf_no = createTree(lc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.rc, leaf_no = createTree(rc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.bestFeature, node.bestValue = bestFeature, bestValue

        # This node is leaf
        if node.lc is None:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)

        return node, leaf_no

    # It is a negative linear relationship
    elif correlation == 'PearsonR(-)':
        node = Node(dataSet)
        bestR = PearsonR(dataSet[:, -2], dataSet[:, -1])
        node.R = bestR
        __slope = stats.linregress(dataSet[:, -2], dataSet[:, -1])[0]
        node.slope = __slope
        if bestR <= -threshold:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)
            return node, leaf_no

        numFeatures = len(dataSet[0]) - 2
        splitSuccess = False
        bestFeature = -1
        bestValue = 0

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = sorted(list(set(featList)))

            for value in range(len(uniqueVals) - 1):
                subDataSetA, subDataSetB = splitDataSet(dataSet, i, uniqueVals[value])

                if subDataSetA.shape[0] <= minsize - 1 or subDataSetB.shape[0] <= minsize - 1:
                    continue

                newRa = PearsonR(subDataSetA[:, -2], subDataSetA[:, -1])
                newRb = PearsonR(subDataSetB[:, -2], subDataSetB[:, -1])

                R = (newRa + newRb) / 2

                if R - bestR <= -mininc:
                    splitSuccess = True
                    bestR = R
                    lc = subDataSetA
                    rc = subDataSetB
                    bestFeature = i
                    bestValue = uniqueVals[value]

        if splitSuccess:
            node.lc, leaf_no = createTree(lc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.rc, leaf_no = createTree(rc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.bestFeature, node.bestValue = bestFeature, bestValue

        if node.lc is None:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)

        return node, leaf_no

    elif correlation == 'MIC':
        node = Node(dataSet)
        bestR = MIC(dataSet[:, -2], dataSet[:, -1])
        node.R = bestR
        node.slope == None
        if bestR >= threshold:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)
            return node, leaf_no

        numFeatures = len(dataSet[0]) - 2
        splitSuccess = False
        bestFeature = -1
        bestValue = 0

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = sorted(list(set(featList)))
            for value in range(len(uniqueVals) - 1):
                subDataSetA, subDataSetB = splitDataSet(dataSet, i, uniqueVals[value])

                if subDataSetA.shape[0] <= minsize - 1 or subDataSetB.shape[0] <= minsize - 1:
                    continue

                newRa = MIC(subDataSetA[:, -2], subDataSetA[:, -1])
                newRb = MIC(subDataSetB[:, -2], subDataSetB[:, -1])

                R = (newRa + newRb) / 2

                if R - bestR >= mininc:
                    splitSuccess = True
                    bestR = R
                    lc = subDataSetA
                    rc = subDataSetB
                    bestFeature = i
                    bestValue = uniqueVals[value]

        if splitSuccess:
            node.lc, leaf_no = createTree(lc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.rc, leaf_no = createTree(rc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.bestFeature, node.bestValue = bestFeature, bestValue

        if node.lc is None:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)

        return node, leaf_no

    elif correlation == 'R2':
        node = Node(dataSet)
        bestR = R2(dataSet[:, -2], dataSet[:, -1])
        node.R = bestR
        node.slope == None
        if bestR >= threshold:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)
            return node, leaf_no

        numFeatures = len(dataSet[0]) - 2
        splitSuccess = False
        bestFeature = -1
        bestValue = 0

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = sorted(list(set(featList)))

            for value in range(len(uniqueVals) - 1):
                subDataSetA, subDataSetB = splitDataSet(dataSet, i, uniqueVals[value])

                if subDataSetA.shape[0] <= minsize - 1 or subDataSetB.shape[0] <= minsize - 1:
                    continue

                newRa = R2(subDataSetA[:, -2], subDataSetA[:, -1])
                newRb = R2(subDataSetB[:, -2], subDataSetB[:, -1])

                R = (newRa + newRb) / 2

                if R - bestR >= mininc:
                    splitSuccess = True
                    bestR = R
                    lc = subDataSetA
                    rc = subDataSetB
                    bestFeature = i
                    bestValue = uniqueVals[value]

        if splitSuccess:
            node.lc, leaf_no = createTree(lc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.rc, leaf_no = createTree(rc, feats, leaf_no, correlation, minsize, threshold, mininc)
            node.bestFeature, node.bestValue = bestFeature, bestValue

        if node.lc is None:
            node.leaf_no = leaf_no
            leaf_no += 1
            write_csv(node, feats)

        return node, leaf_no


# Use graphviz to visualize the TCLR
def render(label, node, dot, feats):
    mark = ''
    if node.slope == None:
        mark = "#=" + str(node.size) + " , ρ=" + str(round(node.R, 3))
    else:
        mark = "#=" + str(node.size) + " , ρ=" + str(round(node.R, 3)) + ' , slope=' + str(round(node.slope, 3))

    if node.lc is None:
        mark = 'No_{}, '.format(node.leaf_no) + mark
    dot.node(label, mark)

    if node.lc is not None:
        render(label + 'A', node.lc, dot, feats)
        render(label + 'B', node.rc, dot, feats)
        dot.edge(label, label + 'A', feats[node.bestFeature] + "≤" + str(node.bestValue))
        dot.edge(label, label + 'B', feats[node.bestFeature] + ">" + str(node.bestValue))


def write_csv(node, feats):
    frame = {}
    for i in range(len(feats)):
        frame[feats[i]] = node.data[:, i]
    frame = pd.DataFrame(frame)
    frame.to_csv('Segmented/subdataset_{}.csv'.format(str(node.leaf_no)))
    return node


def start(filePath, correlation, minsize, threshold, mininc):
    timename = time.localtime(time.time())
    namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min
    csvData = pd.read_csv(filePath)
    feats = [column for column in csvData]
    csvData = np.array(csvData)
    root, _ = createTree(csvData, feats, 0, correlation, minsize, threshold, mininc)

    dot = Digraph(comment='Result of TCLR')
    render('A', root, dot, feats)
    dot.render(
        'Result of TCLR {year}.{month}.{day}-{hour}.{minute}'.format(year=namey, month=nameM, day=named, hour=nameh,
                                                                     minute=namem))

    return True


if __name__ == '__main__':
    start('testdata.csv', 'PearsonR(+)', 3, 0.95, 0.01)
