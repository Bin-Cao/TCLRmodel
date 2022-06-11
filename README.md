# TCLR package
TCLR, Version 1, October, 2021. 

<img width="639" alt="Screen Shot 2022-06-12 at 7 12 57 AM" src="https://user-images.githubusercontent.com/86995074/173207857-795e3a24-af67-4e1a-b80a-4f177712044f.png">


Tree-Classifier for Linear Regression (TCLR) is a novel Machine learning model to capture the functional relationships between features and a target based on correlation.

Reference paper : Cao B, Yang S, Sun A, Dong Z, Zhang TY. Domain knowledge-guided interpretive machine learning - formula discovery for the oxidation behaviour of ferritic-martensitic steels in supercritical water. J Mater Inf 2022. 

Doi: http://dx.doi.org/10.20517/jmi.2022.04

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Installing TCLR
+ pip install TCLR

## Running TCLR

+ from TCLR import TCLRalgorithm as model
+ dataSet = "testdata.csv" # dataset name
+ correlation = 'PearsonR(+)'
+ minsize, threshold, mininc = 3, 0.9, 0.01
+ model.start(dataSet, correlation, minsize, threshold, mininc)


TCLR.V1.0 is coded by Paython language, thus the Integrated Development Environment (IDE) of Python is essential. TCLR can be executed directly through the python IDE, or used by downloading the Graphical User Interfaces (GUI) we provided： TCLR_interface.exe

<u>The GUI can be  download at Releases ——> GUI of TCLR </u>

**graphviz** package is needed for running TCLR, which can be downloaded from the official website http://www.graphviz.org/.

Output: 
+ classification structure tree in pdf format（Result of TCLR.pdf)
+ a folder called 'Segmented' for saving the subdataset of each leaf


<u>The GUI can be  download at Releases ——> GUI of TCLR </u>

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

