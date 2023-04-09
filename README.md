🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks !

Source code : [![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/TCLR/)
# TCLR 


<img width="214" alt="Screen Shot 2022-07-30 at 22 31 40" src="https://user-images.githubusercontent.com/86995074/230752321-e6a1706d-c024-4ae4-8592-02e04516bdf5.png">

## TCLR, Version 1, October, 2021. 

Tree-Classifier for Linear Regression (TCLR) is a novel Machine learning model to capture the functional relationships between features and a target based on correlation. see Template.


TCLR算法通过提供的数据集得到研究变量和时间指数等物理变量之间的显示公式，适用于腐蚀、蠕变等满足动力学或者热力学的物理过程。通过最大化激活能和最小化时间指数可以高效地设计具有高耐腐蚀等优异性能的合金。最新版本V1.4，附有安装说明（用户手册）和运行模版（例子）。


Reference :  
+ (JMI) Cao et al., Doi : http://dx.doi.org/10.20517/jmi.2022.04 
+ (JMST) Wei et al., Doi : https://doi.org/10.1016/j.jmst.2022.11.040


Papers related : [![](https://img.shields.io/badge/Refs-TCLR-yellowgreen)](https://scholar.google.com.hk/scholar?cites=13374282506807262836&as_sdt=2005&sciodt=0,5&hl=zh-CN)


Cite :
+ Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning: formula discovery for the oxidation behavior of ferritic-martensitic steels in supercritical water. J Mater Inf 2022;2:4. http://dx.doi.org/10.20517/jmi.2022.04

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Installing / 安装
    pip install TCLR 
    
## Checking / 查看
    pip show TCLR 
    
## Updating / 更新
    pip install --upgrade TCLR

## Running / 运行
### see Template

``` javascript
#coding=utf-8
from TCLR import TCLRalgorithm as model


dataSet = "testdata.csv"
correlation = 'PearsonR(+)'
minsize = 3
threshold = 0.9
mininc = 0.01
split_tol = 0.8

## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.
model.start(filePath = dataSet, correlation = correlation, minsize = minsize, threshold = threshold,mininc = mininc ,split_tol = split_tol,)

```

### note
``` javascript
:param correlation : {'PearsonR(+)','PearsonR(-)',''MIC','R2'}，default PearsonR(+).
        Methods:
        * PearsonR: (+)(-). for linear relationship.
        * MIC for no-linear relationship.
        * R2 for no-linear relationship.

:param tolerance_list: constraints imposed on features, default is null
        list shape in two dimensions, viz., [['feature_name1',tol_1],['feature_name2',tol_2]...]
        'feature_name1', 'feature_name2' （string） are names of input features;
        tol_1, tol_2 （float, between 0 to 1）are feature's tolerance ratios;
        the variations of feature values on each leaf must be in the tolerance;
        if tol_1 = 0, the value of feature 'feature_name1' must be a constant on each leaf,
        if tol_1 = 1, there is no constraints on value of feature 'feature_name1';
        example: tolerance_list = [['feature_name1',0.2],['feature_name2',0.1]].

:param gpl_dummyfea: dummy features in gpleran regression, default is null
        list shape in one dimension, viz., ['feature_name1','feature_name2',...]
        dummy features : 'feature_name1','feature_name2',... are not used anymore in gpleran regression 

:param minsize : a int number (default=3), minimum unique values for linear features of data on each leaf.

:param threshold : a float (default=0.9), less than or equal to 1, default 0.95 for PearsonR.
        In the process of dividing the dataset, the smallest relevant index allowed in the you research.
        To avoid overfitting, threshold = 0.5 is suggested for MIC 0.5.

:param mininc : Minimum expected gain of objective function (default=0.01)

:param split_tol : a float (default=0.8), constrained features value shound be narrowed in a minmimu ratio of split_tol on split path

:param gplearn : Whether to call the embedded gplearn package of TCLR to regress formula (default=False).

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
+ *Integrated symbolic regression algorithm of gplearn package.
Derive an analytical formula between features and solpes by gplearn*
+ *add a new parameter of tolerance_list, see document*

TCLR V1.5 Aug, 2022.
+ *add a new parameter of gpl_dummyfea, see document*

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

欢迎与我联系，开展交流合作：曹斌（bcao@shu.edu.cn）

