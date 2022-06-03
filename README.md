# TCLR
TCLR, Version 1, October, 2021. 

Tree-Classifier for Linear Regression (TCLR) is a novel tree model to capture the functional relationships between features and a target based on correlation.The entire feature space is partitioned into many sub-spaces, one terminal node representing one sub-space, by hyperplanes parallel to feature axes. In each leaf, the target is linearly proportional to features with certain values of coefficients. The linear relationship is proposed in prior by domain knowledge and theoretically represents a mechanism of studied phenomenon. Furthermore, the obtained values of coefficients shed light on the detail of mechanism.

Reference paper : Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning - formula discovery for the oxidation behaviour of ferritic-martensitic steels in supercritical water. J Mater Inf 2022. 

Doi: http://dx.doi.org/10.20517/jmi.2022.04

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Running TCLR

TCLR.V1.0 is coded by Paython language, thus the Integrated Development Environment (IDE) of Python is essential. TCLR can be executed directly through the python IDE, or used by downloading the Graphical User Interfaces (GUI) we provided： TCLR_interface.exe

<u>The GUI can be  download at Releases ——> GUI of TCLR </u>

**graphviz** package is needed for running TCLR, which can be downloaded from the official website http://www.graphviz.org/.

Output: 
+ classification structure tree in pdf format（Result of TCLR.pdf)
+ a folder called 'Segmented' for saving the subdataset of each leaf


<u>The GUI can be  download at Releases ——> GUI of TCLR </u>
## Contents 
+ Source Code 
+ Template : templates for calling TCLR

## Update log
TCLR V1.1 April, 2022. 
*debug and print out the slopes when Pearson is used*

TCLR V1.2 May, 2022.
*Save the dataset of each leaf*

TCLR V1.3 Jun, 2022.
*Para: minsize - Minimum unique values for linear features of data on each leaf (Minimum number of data on each leaf before V1.3)*

## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

