# Comparision of machine learning algorithms to detect heart disease using scikit learn and python pandas

main project is in FinalProject.ipynb
dataset is aggregated in heart_disease_all15.csv
output of the project can be seen in FinalProject.html



Project Description

Hailing from a healthcare Company. We capture various Electronics Health records (EHR) and various health conditions and Symptoms as a part of the healthcare Delivery. As of now We are only able to achieve the raw Diagnostic report to the customer once we process his samples in the lab. We are not able to do Any value. So, as the Machine learning industry has involved. We would like to go ahead and would like to inform the customer of the potential diseases . So as the Machine learning industry has involved. We would like to go ahead and would like to inform the customer of the potential diseases that he might have based on the previous data we have captured. We can check the condition of the customer and should be able to suggest underlying diseases with certain amount of certainty, that he might have based on the previous data we have captured. We can check the condition of the customer and should be able to suggest underlying diseases with certain amount of certainty.

2.	Scope of Work

There is a large amount of data for clinical assessment of the patients. These different types of data/conditions must be maintained and considered during the diagnosis evaluation process. A need for proper management to extract and process the data efficiently and effectively exists . This can be done using the machine learning algorithms .
Data are divided and analysed in the ML classifiers. This paper focus on different machine learning algorithms used in the medical diagnosis and their advantages and disadvantages. But to choose the right Machine teaming Algorithm to detect a disease is an art. The paper aims at comparing different machine learning algorithms to detect a heart disease with initial parameters. 

Once we are able to find the accuracy of all the algoritms we will work on Finetuning the parameters using GridSearchcv and k-fold cross validation to find out the best possible accuracy for the machine learning Algorithms.

3.	Dataset Description

The dataset has been derived from 4 different datasets and unwanted columns removed: 
1.	Cleveland Clinic Foundation: https://data.world/uci/heart-disease/file/cleveland.data.csv
2.	Hungarian Institute of Cardiology: https://data.world/uci/heart-disease/file/hungarian.data.csv
3.	V.A. Medical Center, Long Beach: https://data.world/uci/heart-disease/file/long-beach-va.data.csv
4.	University Hospital,Zurich, Switzerland: https://data.world/uci/heart-disease/file/switzerland.data.csv

4.Features Description
The table below, summarizes the features information:
FEATURE	DETAIL
1.	AGE	The age of patient in years
2.	GENDER	0 -female, 1 -male
3.	THE CHEST PAIN TYPE	1: typical angina. 2: atypical Angina.  3: non-anginal pain. 4: asymptomatic
4.	THE RESTING BLOOD PRESSURE OF PATIENT	In mm Hg calculated when admitted to the hospital
5.	SERUM CHOLESTEROL	Calculated In mg/dI
6.	IS FASTING BLOOD SUGAR > 120 MG/DI	0 -false , 1- true
7.	RESTING ELECTROCARDIOGRAPHIC RESULTS	0: normal. 1: having ST-T wave abnormalities (T-wave inversions &/or ST elevations or depressions > 0.05 mV).  2: showing probable / definite left ventricular hypertrophy by Estes’s criteria
8.	MAXIMUM HEART RATE ACHIEVED	centered
9.	HAVING EXERCISE-INDUCED ANGINA	1 - yes; 0 - no
10.	ST DEPRESSION INDUCED BY EXERCISE RELATIVE TO REST	In mm Hg calculated when admitted to the hospital
11.	THE NUMBER OF MAJOR VESSELS	(0-3) Color developed by fluoroscopy
12.	THE SLOPE OF THE PEAK EXERCISE ST-SEGMENT	Value1: upsloping. Value2: flat. Value3: downsloping
13.	THALLIUM HEART SCAN	3: normal,  6: fixed defect,  7: reversible defect
14.	HEART DISEASE DIAGNOSIS (CORONARY HEART DISEASE)	Value 0 : disease not present. Value1: heart disease


4.	Objectives

Machine learning is a process which learns from past experiences to improve future performances. Aim of this field is to automate learning methodologies. Medical treatments start with screening of patient’s health symptoms, conditional diagnosis, treatment, and frequent monitoring. Precise diagnosis is very important in deciding the right therapy at an early stage. But in multiple cases it is extremely difficult for the experts to identify the level of the patient. With the existing clinical records, the ML methods can be used to produce various descriptive analysis of clinical features. Machine Learning algorithms are being widely used for diagnosis of several diseases like diabetics, heart problems, cancer.

There are various machine learning techniques for disease detection:

1. Supervised Learning: The popular algorithms of Supervise learning are classification and regression. The supervised learning is of two types : the parametric models and the non-parametric models.
a. In the parametric models, the predictive function is a combination of fixed no of parameters. First stage is the learning stage using the training dataset. After this stage the training data can be discarded as the prediction for new input is based on the learned parameters. Some examples of parametric models are Linear regression and classification ,neural networks.

b. In nonparametric models, the no of parameters is dependent on the training dataset. The training data set is maintained for the prediction. The most commonly used nonparametric models are support vector machines and nearest neighbour algorithm.

2. Semi supervised learning: This is a combination of the supervised and the unsupervised machine learning techniques. It is an repetitive process where initially the labeled data values are used with supervised learning and next the unlabelled data values are labelled with the previous known values and finally all the data values are used for predicting the outcome. Basically, the semi-supervised learning models can be categorized under four classes: a) generative model b) model where the decision boundary is in low-density region c) graph-based model d) two-step model with unsupervised learning followed by supervised learning.

3. Unsupervised Learning: These algorithms predicts the result based on  similarity in the input Data. Clustering is one of the unsupervised learning used generally. It is the method of grouping the similar data based on their distance. The property of clusters is that there should be a high intra-cluster similarity and a low inter-cluster similarity. The detected clusters depend on the input data values.

4. Deep learning: It is another form of machine learning that has a high-level Abstraction.

