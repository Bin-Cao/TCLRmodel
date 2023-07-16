🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks !

if you have any questions or need help, you are welcome to contact me

Source code : [![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/TCLR/)
# TCLR 

<img width="214" alt="Screen Shot 2022-07-30 at 22 31 40" src="https://user-images.githubusercontent.com/86995074/230752321-e6a1706d-c024-4ae4-8592-02e04516bdf5.png">

## Package Document / 手册
see 📒 [TCLR](https://tclr.netlify.app) (Click to view）


By incorporating the TCLR into a thermodynamic kinetic framework, it becomes possible to accurately predict the rates of chemical reactions as a function of temperature, pressure, and other system variables. This can be useful in a variety of fields, including materials science, chemical engineering, and biochemistry, where accurate modeling of reaction rates is essential for the design and optimization of chemical processes.


TCLR算法通过提供的数据集得到研究变量和时间指数等物理变量之间的显示公式，适用于腐蚀、蠕变等满足动力学或者热力学的物理过程。通过最大化激活能和最小化时间指数可以高效地设计具有高耐腐蚀等优异性能的合金。附有安装说明（用户手册）和运行模版（例子）。


<img width="1044" alt="Screenshot 2023-04-09 at 11 34 48" src="https://user-images.githubusercontent.com/86995074/230752794-54903df8-030b-4630-a1aa-5b67a135c0e5.png">


Cite :
+ Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning: formula discovery for the oxidation behavior of ferritic-martensitic steels in supercritical water. J Mater Inf 2022;2:4. http://dx.doi.org/10.20517/jmi.2022.04

Reference :  
+ (JMI) Cao et al., Doi : http://dx.doi.org/10.20517/jmi.2022.04 
+ (JMST) Wei et al., Doi : https://doi.org/10.1016/j.jmst.2022.11.040


Papers related : [![](https://img.shields.io/badge/Refs-TCLR-yellowgreen)](https://scholar.google.com.hk/scholar?cites=13374282506807262836&as_sdt=2005&sciodt=0,5&hl=zh-CN)


### The formular we proposed in the paper
``` javascript
import numpy as np

def FMO_formular(Cr, T=673.15, t = 600, DOC = 10):

    """
    Cao B, Yang S, Sun A, Dong Z, Zhang TY. 
    Domain knowledge-guided interpretive machine learning: 
    formula discovery for the oxidation behavior of ferritic-martensitic 
    steels in supercritical water. J Mater Inf 2022;2:4. 
    http://dx.doi.org/10.20517/jmi.2022.04
    
    input:
    Cr : oxidation chromium equivalent concentration (wt.%), 10.38 <= Cr <= 30.319
    Cr = [Cr] + 40.3[V] + 2.3[Si] + 10.7[Ni] − 1.5[Mn]
    T : Absolute temperature (K), 673.15 <= T <= 923.15
    t : Exposure time (h), 30 <= t <= 2000
    DOC : Dissolved oxygen concentration (ppb), 0 <= DOC <= 8000
    
    output:
    the logarithm of weight gain (mg / dm2)
    """

    # Eq.(6c) in paper
    pre_factor = 0.084*(Cr**3/(T-DOC) - np.sqrt(T+DOC)) + 0.98*(Cr-DOC/T) / np.log(Cr+DOC)+8.543
    
    # Eq.(5a) in paper
    Q = 0.084*(Cr**2-Cr+DOC) / np.exp(DOC/T) + 45.09
    
    # Eq.(5b) in paper
    m = 0.323 - 0.061 * np.exp(DOC/T) / (Cr - np.sqrt(Cr) - DOC)
    
    ln_wg = pre_factor + np.log(DOC+2.17) -  Q * 1000 / 8.314 / T + m*np.log(t)
    
    return ln_wg
```    

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

model.start(filePath = dataSet, correlation = correlation, minsize = minsize, threshold = threshold,
            mininc = mininc ,split_tol = split_tol,)

```


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
+ *add a new parameter of tolerance_list*

TCLR V1.5 Aug, 2022.
+ *add a new parameter of gpl_dummyfea*

TCLR Apr, 2023.
+ *user documentation*
+ *web interface*

## About / 更多
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.

