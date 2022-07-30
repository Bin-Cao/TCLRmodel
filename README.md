
[![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/TCLR/)
# TCLR 

![WechatIMG65](https://user-images.githubusercontent.com/86995074/181872169-15e4f6fd-87ad-4a76-81f9-21de1a8e7012.jpeg =250x)


## TCLR, Version 1, October, 2021. 

Tree-Classifier for Linear Regression (TCLR) is a novel Machine learning model to capture the functional relationships between features and a target based on correlation.

TCLR算法通过提供的数据集得到研究变量和时间指数等物理变量之间的显示公式，适用于腐蚀、蠕变等满足动力学或者热力学的物理过程。通过最大化激活能和最小化时间指数可以高效地设计具有高耐腐蚀等优异性能的合金。最新版本V1.4，附有安装说明（用户手册）和运行模版（例子）。

Reference paper : Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning - formula discovery for the oxidation behaviour of ferritic-martensitic steels in supercritical water. J Mater Inf 2022. 

Doi : http://dx.doi.org/10.20517/jmi.2022.04

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Installing / 安装
    pip install TCLR 

## Updating / 更新
    pip install --upgrade TCLR

## Running / 运行
### Ref. https://github.com/Bin-Cao/TCLRmodel/tree/main/Source%20Code

``` javascript
# coding=utf-8

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
    Reference :  DOI : 10.20517/jmi.2022.04
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

```

output 运行结果: 
+ classification structure tree in pdf format（Result of TCLR.pdf) 图形结果
+ a folder called 'Segmented' for saving the subdataset of each leaf (passed test) 数据文件

note 注释: 

the complete execution template can be downloaded at the *Example* folder 算法运行模版可在 *Example* 文件夹下载

**graphviz** (recommended installation) package is needed for generating the graphical results, which can be downloaded from the official website http://www.graphviz.org/. see user guide.（推荐安装）用于生成TCLR的图形化结果, 下载地址: http://www.graphviz.org/.


## Update log / 日志
TCLR V1.1 April, 2022. 
*debug and print out the slopes when Pearson is used*

TCLR V1.2 May, 2022.
*Save the dataset of each leaf*

TCLR V1.3 Jun, 2022.
*Para: minsize - Minimum unique values for linear features of data on each leaf (Minimum number of data on each leaf before V1.3)*

TCLR V1.4 Jun, 2022.
*Integrated symbolic regression algorithm of gplearn package.
Derive an analytical formula between features and solpes by gplearn*

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

