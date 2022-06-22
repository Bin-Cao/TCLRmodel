# TCLR package 
TCLR, Version 1, October, 2021. 

![WechatIMG1262](https://user-images.githubusercontent.com/86995074/174769421-de1a3a51-2ff4-431d-9bb5-66b5d4922c76.png)


Tree-Classifier for Linear Regression (TCLR) is a novel Machine learning model to capture the functional relationships between features and a target based on correlation.

TCLR算法通过提供的数据集得到研究变量和时间指数等物理变量之间的显示公式，适用于腐蚀、蠕变等满足动力学或者热力学的物理过程。通过最大化激活能和最小化时间指数可以高效地设计具有高耐腐蚀等优异性能的合金。最新版本V1.4，附有安装说明和运行模版。

Reference paper : Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning - formula discovery for the oxidation behaviour of ferritic-martensitic steels in supercritical water. J Mater Inf 2022. 

Doi: http://dx.doi.org/10.20517/jmi.2022.04

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Installing / 安装
+ pip install TCLR

## Updating / 更新
+ pip install --upgrade TCLR

## Running / 运行

+ from TCLR import TCLRalgorithm as model
+ dataSet = "testdata.csv" # dataset name
+ correlation = 'PearsonR(+)'
+ minsize, threshold, mininc = 3, 0.9, 0.01
+ model.start(dataSet, correlation, minsize, threshold, mininc, gplearn = True)


Output 运行结果: 
+ classification structure tree in pdf format（Result of TCLR.pdf) 图形结果
+ a folder called 'Segmented' for saving the subdataset of each leaf (passed test) 数据文件

note : the complete execution template can be downloaded at the *Source Code** folder 算法运行模版可在 *Source Code* 文件夹下载


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
Derive an analytical formula between features and solpes by symbolic regression*

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

