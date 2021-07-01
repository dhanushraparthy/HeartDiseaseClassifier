# Libraries
import numpy as np
from flask import *
import pickle

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV


#!/usr/bin/env python
# coding: utf-8

# # Detailed Comparison of Machine learning and Processing Techniques for Disease Detection
# 
# Cardiovascular diseases are the leading cause of death globally, resulted in 17.9 million deaths (32.1%) in 2015, up from 12.3 million (25.8%) in 1990. It is estimated that 90% of CVD is preventable. There are many risk factors for heart diseases that we will take a closer look at.
# 
# The main objective of this study is to build a model that can predict the heart disease occurrence, based on a combination of features (risk factors) describing the disease. Different machine learning classification techniques will be implemented and compared upon standard performance metric such as accuracy.
# 
# The dataset used for this study was taken from UCI machine learning repository, titled “Heart Disease Data Set”.
# 
# Contents of the Notebook:
# 
# Dataset structure & description
# Analyze, identify patterns, and explore the data
# Data preparation
# Modelling and predicting with Machine Learning
# Conclusion
# 

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib

from sklearn.metrics import accuracy_score,roc_auc_score
import scikitplot as skplt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# visualization
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # 1. Dataset structure & description
# The dataset used in this project contains 14 variables. The independent variable that needs to be predicted, 'diagnosis', determines whether a person is healthy or suffer from heart disease. Experiments with the Cleveland database have concentrated on endeavours to distinguish disease presence (values 1, 2, 3, 4) from absence (value 0). There are several missing attribute values, distinguished with symbol '?'. The header row is missing in this dataset, so the column names have to be inserted manually.
# 
# Features information:
# age - age in years
# sex - sex(1 = male; 0 = female)
# chest_pain - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
# blood_pressure - resting blood pressure (in mm Hg on admission to the hospital)
# serum_cholestoral - serum cholestoral in mg/dl
# fasting_blood_sugar - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# electrocardiographic - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
# max_heart_rate - maximum heart rate achieved
# induced_angina - exercise induced angina (1 = yes; 0 = no)
# ST_depression - ST depression induced by exercise relative to rest
# slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
# no_of_vessels - number of major vessels (0-3) colored by flourosopy
# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# diagnosis - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)
# Types of features:
# Categorical features (Has two or more categories and each value in that feature can be categorised by them): sex, chest_pain
# 
# Ordinal features (Variable having relative ordering or sorting between the values): fasting_blood_sugar, electrocardiographic, induced_angina, slope, no_of_vessels, thal, diagnosis
# 
# Continuous features (Variable taking values between any two points or between the minimum or maximum values in the feature column): age, blood_pressure, serum_cholestoral, max_heart_rate, ST_depression
# 
# 

app = Flask(__name__)

col_names = ['age','sex','chest_pain','blood_pressure','serum_cholestoral','fasting_blood_sugar', 'electrocardiographic',
				 'max_heart_rate','induced_angina','ST_depression','slope','no_of_vessels','thal','diagnosis']

@app.route('/loaddataset')
def loaddataset():
	# column names in accordance with feature information
	# read the file
	df = pd.read_csv("heart_disease_all15.csv", names=col_names, header=None, na_values="?")
	data1= ("Number of records: {}\nNumber of variables: {}".format(df.shape[0], df.shape[1]), 		df.head(), df.info() )
	return render_template('loaddataset.html', 
		data11=data1[0], data12=data1[1].to_html(header='true', classes='table table-striped table-hover table-success table-bordered table-responsive'), data13=data1[2])


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import io


@app.route('/daPlot2.png')
def daPlot2():
	fig = Figure()
	ax=[]
	ax.append(fig.add_subplot(1, 2, 1))
	ax.append(fig.add_subplot(1, 2, 2))
	#f, ax = plt.subplots(1,2,figsize=(14,6))
	dfile=open('df','rb')
	df = pickle.load(dfile)
	dfile.close()

	df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
	ax[0].set_title('diagnosis')
	ax[0].set_ylabel('')
	sns.countplot('diagnosis', data=df, ax=ax[1])
	#plt.show()
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')
	
@app.route('/daPlot3.png')
def daPlot3():
	fig = Figure()
	ax=[]
	ax.append(fig.add_subplot(2, 2, 1))
	ax.append(fig.add_subplot(2, 2, 2))
	ax.append(fig.add_subplot(2, 2, 3))
	ax.append(fig.add_subplot(2, 2, 4))
	#plt.figure(figsize=(12,10))
	#plt.subplot(221)
	dfile=open('df','rb')
	df = pickle.load(dfile)
	dfile.close()

	sns.distplot(df[df['diagnosis']==0].age, ax=ax[0])
	ax[0].set_title('Age of patients without heart disease')
	#plt.subplot(222)
	sns.distplot(df[df['diagnosis']==1].age, ax=ax[1])
	ax[0].set_title('Age of patients with heart disease')
	#plt.subplot(223)
	sns.distplot(df[df['diagnosis']==0].dropna().dropna().max_heart_rate, ax=ax[2])
	ax[0].set_title('Max heart rate of patients without heart disease')
	#plt.subplot(224)
	sns.distplot(df[df['diagnosis']==1].dropna().max_heart_rate, ax=ax[3])
	ax[0].set_title('Max heart rate of patients with heart disease')
	#plt.show()
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')
	
@app.route('/dataAnalysis')
def dataAnalysis():
	#df = pd.read_json(session['df'])
	df = pd.read_csv("heart_disease_all15.csv", names=col_names, header=None, na_values="?")
	# In[4]:
	# extract numeric columns and find categorical ones
	numeric_columns = ['serum_cholestoral', 'max_heart_rate', 'age', 'blood_pressure', 'ST_depression']
	categorical_columns = [c for c in df.columns if c not in numeric_columns]
	print(categorical_columns)
	# # 2.Analyze features, identify patterns, and explore the data¶
	# Target value
	# Knowing the distribution of target value is vital for choosing appropriate accuracy metrics and consequently properly assess different machine learning models.
	# 
	# Since the values 1-4 indicate that a disease is present, it's reasonable to pull them together.
	# In[5]:
	# count values of explained variable
	# df.diagnosis.value_counts()
	# In[6]:
	print(df.diagnosis)
	df.diagnosis = (df.diagnosis != 0).astype(int)	
	print(df.diagnosis)
	dfile=open('df','wb')
	pickle.dump(df, dfile)
	dfile.close()
	# df = pickle.load('df')
	return render_template('dataanalysis.html', 
		catcols=categorical_columns, diagcount=df.diagnosis.value_counts())	

@app.route('/dpPlot1.png')
def dpPlot1():
	dfile=open('df','rb')
	df = pickle.load(dfile)
	dfile.close()

	fig = Figure()
	ax=[]
	ax.append(fig.add_subplot(1, 3, 1))
	ax.append(fig.add_subplot(1, 3, 2))
	ax.append(fig.add_subplot(1, 3, 3))

	# create pairplot and two barplots
	sns.pointplot(x="sex", y="diagnosis", hue='chest_pain', data=df, ax=ax[0])
	#ax[0].legend(['male = 1', 'female = 0'])
	sns.barplot(x="induced_angina", y="diagnosis", data=df, ax=ax[1])
	#ax[1].legend(['yes = 1', 'no = 0'])
	sns.countplot(x="slope", hue='diagnosis', data=df, ax=ax[2])

	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')


@app.route('/dataPrepare')
def dataPrepare():

	dfile=open('df','rb')
	df = pickle.load(dfile)
	dfile.close()

	# # Categorical features
	# Let's take a closer look at categorical variables and see how they impact our target.
	# In[11]:
	# count ill vs healthy people grouped by sex
	df.groupby(['sex','diagnosis'])['diagnosis'].count()
	# In[12]:
	# average number of diagnosed people grouped by number of blood vessels detected by fluoroscopy
	df[['no_of_vessels','diagnosis']].dropna().where(df['no_of_vessels']>= 0).groupby('no_of_vessels').count().apply(lambda x:
													 100 * x / float(x.sum()))


	# # In[13]:
	# # create pairplot and two barplots
	# plt.figure(figsize=(16,6))
	# plt.subplot(131)
	# sns.pointplot(x="sex", y="diagnosis", hue='chest_pain', data=df)
	# plt.legend(['male = 1', 'female = 0'])
	# plt.subplot(132)
	# sns.barplot(x="induced_angina", y="diagnosis", data=df)
	# plt.legend(['yes = 1', 'no = 0'])
	# plt.subplot(133)
	# sns.countplot(x="slope", hue='diagnosis', data=df)
	# plt.show()

	# # Observations:
	# Men are much more prone to get a heart disease than women.
	# The higher number of vessels detected through fluoroscopy, the higher risk of disease.
	# While soft chest pain may be a bad symptom of approaching problems with heart (especially in case of men), strong pain is a serious warning!
	# Risk of getting heart disease might be even 3x higher for someone who experienced exercise-induced angina.
	# The flat slope (value=2) and downslope (value=3) of the peak exercise indicates a high risk of getting diseas

	# # 3.Data Preparation
	# In order to make our dataset compatible with machine learning algorithms contained in Sci-kit Learn library, first of all, we need to handle all missing data.
	# 
	# There are many options we could consider when replacing a missing value, for example:
	# 
	# A constant value that has meaning within the domain, such as 0, distinct from all other values
	# A value from another randomly selected record
	# A mean, median or mode value for the column
	# A value estimated by another predictive model

	# fill missing values with mode
	df.dropna(how='any', inplace=True) 
	df = df[df.no_of_vessels != -9.0]

	dfile=open('df','wb')
	pickle.dump(df, dfile)
	dfile.close()

	return render_template('dataPrepare.html')	


@app.route('/splitData')
def splitData():
	dfile=open('df','rb')
	df = pickle.load(dfile)
	dfile.close()

	# split the data
	X, y = df.iloc[:, :-1].fillna(df.mean()), df.iloc[:, -1]
	X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=2606)

	# Data needs to be normalized or standardized before applying to machine learning algorithms. Standardization scales the data and gives information on how many standard deviations the data is placed from its mean value. Effectively, the mean of the data (µ) is 0 and the standard deviation (σ) is 1.

	# scale feature matrices
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	
	dfile=open('X_train','wb')
	pickle.dump(X_train, dfile)
	dfile.close()
	
	dfile=open('X_test','wb')
	pickle.dump(X_test, dfile)
	dfile.close()
	
	dfile=open('y_train','wb')
	pickle.dump(y_train, dfile)
	dfile.close()
	
	dfile=open('y_test','wb')
	pickle.dump(y_test, dfile)
	dfile.close()	

	return render_template('splitData.html', 
		X_train=pd.DataFrame(X_train).to_html(), 
		X_test=pd.DataFrame(X_test).to_html(), 
		y_train=pd.DataFrame(y_train).to_html(), 
		y_test=pd.DataFrame(y_test).to_html() )	


#	def modellingPredicting():

# # 4. Modelling and predicting with Machine Learning
# The main goal of the entire project is to predict heart disease occurrence with the highest accuracy. In order to achieve this, we will test several classification algorithms. This section includes all results obtained from the study and introduces the best performer according to accuracy metric. I have chosen several algorithms typical for solving supervised learning problems throughout classification methods.
# 
# First of all, let's equip ourselves with a handy tool that benefits from the cohesion of SciKit Learn library and formulate a general function for training our models. The reason for displaying accuracy on both, train and test sets, is to allow us to evaluate whether the model overfits or underfits the data (so-called bias/variance tradeoff).




def train_model_info(X_train, y_train, X_test, y_test, classifier, **kwargs):
	
	"""
	Fit the chosen model and print out the score.
	
	"""
	
	# instantiate model
	model = classifier(**kwargs)
	
	# train model
	model.fit(X_train,y_train)
	
	# check accuracy and print out the results
	fit_accuracy = model.score(X_train, y_train)
	test_accuracy = model.score(X_test, y_test)
	y_pred=model.predict(X_test)
	print(f"Train accuracy: {fit_accuracy:0.2%}")
	print(f"Test accuracy: {test_accuracy:0.2%}")
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
	
	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()
	print(model)
	
	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('Model \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	return model
	
def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
	
	"""
	Fit the chosen model and print out the score.
	
	"""
	
	# instantiate model
	model = classifier(**kwargs)
	
	# train model
	model.fit(X_train,y_train)
	
	# check accuracy and print out the results
	fit_accuracy = model.score(X_train, y_train)
	test_accuracy = model.score(X_test, y_test)
	y_pred=model.predict(X_test)
	print(f"Train accuracy: {fit_accuracy:0.2%}")
	print(f"Test accuracy: {test_accuracy:0.2%}")
	
	return model


@app.route('/knnAlgor')
def knnAlgor():

	#X_train,X_test, y_train,y_test
	
	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()
	

	# # K-Nearest Neighbours (KNN)
	# K-Nearest Neighbors algorithm is a non-parametric method used for classification and regression. The principle behind nearest neighbour methods is to find a predefined number of training samples closest in distance to the new point and predict the label from these.
	# KNN
	model = train_model_info(X_train, y_train, X_test, y_test, KNeighborsClassifier)
	# Seek optimal 'n_neighbours' parameter
	for i in range(1,10):
		print("n_neigbors = "+str(i))
		train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=i)


	# # kNN with grid search
	# Improving the accuracy scores with grid search

	


	from sklearn.model_selection import GridSearchCV
	k_range = list(range(1, 31))
	clf=KNeighborsClassifier()
	clf.fit(X_train,y_train)

	param_grid = dict(n_neighbors=k_range)
	#print(param_grid)
	grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', return_train_score=False,iid=False)
	c=grid.fit(X_train,y_train)
	print(c)
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})

	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score','mean_fit_time', 'params']]




	clf=KNeighborsClassifier(algorithm='auto', leaf_size=30,
												metric='minkowski',
												metric_params=None, n_jobs=None,
												n_neighbors=7, p=2,
												weights='uniform')
	clf.fit(X_train,y_train)


	y_pred=clf.predict(X_test)
	acc=accuracy_score(y_test,y_pred)
	print("Accuracy score {}".format(acc))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))
	
	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()

	print(clf)

	dfile=open('clfKnn','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()

	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

@app.route('/algorPlot.png')
def algorPlot():
	dfile=open('cmat','rb')
	cmat = pickle.load(dfile)
	dfile.close()

	fig = Figure()
	ax=[]
	ax.append(fig.add_subplot(1, 1, 1))

	sns.heatmap(cmat, annot=True, ax=ax[0])
	
	#ax[0].set_title('KNN ')
	#plt.ylabel('True label')
	#plt.xlabel('Predicted label')

	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')
	
@app.route('/decAlgor')
def decisionTreesAlgor():
	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()

	# # Decision Trees
	# DT algorithm creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. It is simple to understand and interpret and it's possible to visualize how important a particular feature was for our tree.


	# Decision Tree
	model = train_model_info(X_train, y_train, X_test, y_test, DecisionTreeClassifier, random_state=None)

	# plot feature importances
	###pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh()


	# Variable 'thal' turns out to be a significantly important feature.
	# Remember my hypothesis that 'fasting_blood_sugar" is a very weak feature? Above graph confirms this clearly.
	# Decision tree model learns the train set perfectly, and at the same time is entirely overfitting the data, what results in poor prediction. Other values of 'max_depth' parameter need to be tried out.

	# In[25]:


	# Check optimal 'max_depth' parameter
	for i in range(1,8):
		print("max_depth = "+str(i))
		train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, max_depth=i, random_state=None)


	# In[26]:


	param_grid = { 'criterion':['gini','entropy'],'min_samples_split' : range(10,500,20),'max_depth': np.arange(1, 15)}
	# decision tree model
	clf=DecisionTreeClassifier()
		#use gridsearch to test all values
	grid = GridSearchCV(clf, param_grid, cv=5,scoring='accuracy')
		#fit model to data
	c=grid.fit(X_train,y_train)
	print(c.best_estimator_)
	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score','mean_fit_time', 'params']]


	# ### Decision Trees with gridcv

	# In[27]:


	clf=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
						   max_features=None, max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=1, min_samples_split=10,
						   min_weight_fraction_leaf=0.0, presort=False,
						   random_state=2606, splitter='best')
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))

	#print(clf.best_estimator_)



	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()
	
	dfile=open('clfDec','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()



	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('Decision Trees Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

@app.route('/logAlgor')
def logisticRegressionAlgor():
	# ## Logistic Regression
	# Logistic regression is a basic technique in statistical analysis that attempts to predict a data value based on prior observations. A logistic regression algorithm looks at the relationship between a dependent variable and one or more dependent variables.

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()


	# Logistic Regression
	model = train_model_info(X_train, y_train, X_test, y_test, LogisticRegression)


	# # logistic regression grid search cv

	# In[29]:


	# Create regularization penalty space
	penalty = ['l2']

	# Create regularization hyperparameter space
	C=[0.001,.009,0.1,.09,0.12,5,10,25,0.2,0.15]
	solver= solver= [ 'lbfgs', 'liblinear', 'sag', 'saga']
	# Create hyperparameter options
	hyperparameters = dict(C=C, penalty=penalty,solver=solver)
	grid = GridSearchCV(LogisticRegression(max_iter=1000), hyperparameters,scoring='accuracy')

	# fitting the model for grid search 
	grid.fit(X_train, y_train) 

	print(grid.best_params_) 

	print(grid.best_estimator_)
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})

	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'mean_fit_time','params']]


	# In[31]:


	clf=LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
					   intercept_scaling=1, max_iter=1000,
					   multi_class='warn', n_jobs=None, penalty='l2',
					   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
					   warm_start=False)


	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))




	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()

	print(clf)

	dfile=open('clfLog','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()

	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('LogisticRegression Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})
	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

def gaussianNaiveBayesAlgor():
	# ## Gaussian Naive Bayes
	# In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()


	#Gaussian Naive Bayes
	model = train_model_info(X_train, y_train, X_test, y_test, GaussianNB)


	# # Support Vector Machines
	# Support Vector Machines are perhaps one of the most popular machine learning algorithms. They are the go-to method for a high-performing algorithm with a little tuning. At first, let's try it on default settings.

	# In[33]:


	# Support Vector Machines
	model = train_model_info(X_train, y_train, X_test, y_test, SVC)


	# # SVM with gridsearch

	# In[34]:


	k_range = list(range(1, 31))


	# defining parameter range 
	param_grid = {'C': [0.1, 1, 10, 100, 0.05],  
				  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
				  'kernel': ['rbf','linear']}  
	  
	grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,scoring='accuracy') 
	  
	# fitting the model for grid search 
	grid.fit(X_train, y_train) 

	print(grid.best_params_) 

	print(grid.best_estimator_)
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})

	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score','mean_fit_time' ,'params']]


	# In[35]:


	print(grid.best_estimator_)

@app.route('/svmAlgor')
def svmAlgor():
	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()

	from sklearn.svm import SVC

	clf=SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False)

	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))




	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()

	print(clf)

	dfile=open('clfSvm','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()

	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

@app.route('/ranAlgor')
def randomForestsAlgor():
	# ## Random Forests
	# Random forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()


	# Random Forests
	model = train_model_info(X_train, y_train, X_test, y_test, RandomForestClassifier, random_state=2606)
	#pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh()


	# # Random Forest with grid

	# In[38]:


	rfc=RandomForestClassifier(random_state=42)

	param_grid = { 
		'n_estimators': [60,100,200, 500],
		'max_features': ['auto', 'sqrt', 'log2'],
		'max_depth' : [4,5,6,7,8],
		'criterion' :['gini', 'entropy']
	}

	grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


	# fitting the model for grid search 
	grid.fit(X_train, y_train) 

	print(grid.best_params_) 

	print(grid.best_estimator_)
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})

	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score','mean_fit_time', 'params']]


	# In[39]:


	#Random Forest Classifier
	
	clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
						   max_depth=4, max_features='auto', max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=1, min_samples_split=2,
						   min_weight_fraction_leaf=0.0, n_estimators=500,
						   n_jobs=None, oob_score=False, random_state=42, verbose=0,
						   warm_start=False)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))


	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()

	print(clf)

	dfile=open('clfRan','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()

	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('Random Forest Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

@app.route('/graAlgor')
def gradientBoostringAlgor():
	# # Gradient Boosting

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()


	# Gradient Boosting
	model = train_model_info(X_train, y_train, X_test, y_test, GradientBoostingClassifier, random_state=2606)


	# ## Gradient Boosting Classifier grid search

	# In[42]:


	param_test2 = {'n_estimators':[40,50,60],'max_depth':range(3,16,1),'learning_rate':[0.2], 'min_samples_split':range(2,500,50),'criterion':['friedman_mse']}

	estimator=GradientBoostingClassifier( max_leaf_nodes=None,  min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,  min_weight_fraction_leaf=0.0,    n_iter_no_change=None, presort='auto',  random_state=None, subsample=1.0, tol=0.0001, warm_start=False, 
							   validation_fraction=0.1, verbose=0)

	grid = GridSearchCV(estimator =estimator,param_grid = param_test2, scoring='accuracy',n_jobs=4,iid=False, cv=5)
	grid.fit(X_train,y_train)
	# fitting the model for grid search 
	#grid.fit(X_train, y_train) 

	  


	print(grid.best_params_) 

	print(grid.best_estimator_)
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})

	pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'mean_fit_time','params']]


	# In[43]:


	#Gradient Boosting Classifier
	#clf=GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
	clf=GradientBoostingClassifier(criterion='friedman_mse', init=None,
							   learning_rate=0.2, loss='deviance', max_depth=11,
							   max_features=None, max_leaf_nodes=None,
							   min_impurity_decrease=0.0, min_impurity_split=None,
							   min_samples_leaf=1, min_samples_split=52,
							   min_weight_fraction_leaf=0.0, n_estimators=40,
							   n_iter_no_change=None, presort='auto',
							   random_state=None, subsample=1.0, tol=0.0001,
							   validation_fraction=0.1, verbose=0,
							   warm_start=False)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print(clf)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))



	confusion=confusion_matrix(y_test, y_pred)
	confusion=np.asarray(confusion)
	print(confusion/confusion.sum())
	cmat=confusion/confusion.sum()

	print(clf)

	dfile=open('clfGra','wb')
	pickle.dump(clf, dfile)
	dfile.close()
	
	dfile=open('cmat','wb')
	pickle.dump(cmat, dfile)
	dfile.close()

	# plt.figure(figsize=(5.5,4))
	# sns.heatmap(cmat, annot=True)
	# plt.title('Gradient Boosting Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
	# plt.ylabel('True label')
	# plt.xlabel('Predicted label')
	# plt.show()
	#pd.DataFrame(data={"Y_Actual":y_test,"Y_Predict":y_pred})
	acc=accuracy_score(y_test,y_pred)
	return render_template('algor.html', acc=acc, clf=clf)

@app.route('/compareAlgors')
def compareAlgors():

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()

	# initialize an empty list
	accuracy = []
	roc=[]
	mean_squared_errors=[]
	precision_scores=[]

	# list of algorithms names
	classifiers = ['KNN', 'Decision Trees', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forests','Gradient Boosting']

	# list of algorithms with parameters
	models = [KNeighborsClassifier(),DecisionTreeClassifier(), LogisticRegression(), GaussianNB(),SVC(),RandomForestClassifier(),GradientBoostingClassifier()]

	# loop through algorithms and append the score into the list
	from sklearn.metrics import precision_score
	from sklearn.metrics import mean_squared_error
	for i in models:
		model = i
		model.fit(X_train, y_train)
		score = model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		accuracy.append(score)
		roc.append(roc_auc_score(y_test,y_pred))
		precision_scores.append(precision_score(y_test, y_pred))
		mean_squared_errors.append(mean_squared_error(y_test, y_pred))
	# create a dataframe from accuracy results
	# create a dataframe from accuracy results
	summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       
	return render_template('compareAlgors.html', summary=summary.to_html(header=True))


@app.route('/predict', methods=['GET'])
def predict():

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()

	# initialize an empty list
	preds = []
	roc=[]
	mean_squared_errors=[]
	precision_scores=[]
	
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	col_names = ['age','gender','chest_pain','blood_pressure','serum_cholestoral','fasting_blood_sugar', 'electrocardiographic',
				 'max_heart_rate','induced_angina','ST_depression','slope','no_of_vessels','thal']
	data = []
	for col in col_names:
		data.append( float( request.args.get(col) ) )
	testdata = [data]
	X_test = scaler.transform(testdata)


	# list of algorithms names
	classifiers = ['KNN', 'Decision Trees', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forests','Gradient Boosting']

	# list of algorithms with parameters
	models = [KNeighborsClassifier(),DecisionTreeClassifier(), LogisticRegression(), GaussianNB(),SVC(),RandomForestClassifier(),GradientBoostingClassifier()]

	# loop through algorithms and append the score into the list
	from sklearn.metrics import precision_score
	from sklearn.metrics import mean_squared_error
	for i in models:
		model = i
		model.fit(X_train, y_train)
		y_pred=model.predict(testdata)
		preds.append(y_pred)
	summary = pd.DataFrame({'preds':preds}, index=classifiers)       
	return render_template('predict.html', summary=summary.to_html(header=True))

def conclusionAlgors():

	# ## 5. Conclusion
	# The goal of the project was to compare different machine learning algorithms and predict 
	# if a certain person, given various personal characteristics and symptoms, will get heart disease or not. 
	# Here are the final results

	dfile=open('X_train','rb')
	X_train = pickle.load(dfile)
	dfile.close()

	dfile=open('X_test','rb')
	X_test = pickle.load(dfile)
	dfile.close()

	dfile=open('y_train','rb')
	y_train = pickle.load(dfile)
	dfile.close()

	dfile=open('y_test','rb')
	y_test = pickle.load(dfile)
	dfile.close()


	# initialize an empty list
	accuracy = []
	roc=[]
	mean_squared_errors=[]

	# list of algorithms names
	classifiers = ['KNN', 'Decision Trees', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forests','Gradient Boosting']

	# list of algorithms with parameters
	models = [KNeighborsClassifier(algorithm='auto', leaf_size=30,
												metric='minkowski',
												metric_params=None, n_jobs=None,
												n_neighbors=7, p=2,
												weights='uniform'), DecisionTreeClassifier(class_weight=None,
												  criterion='gini', max_depth=None,
												  max_features=None,
												  max_leaf_nodes=None,
												  min_impurity_decrease=0.0,
												  min_impurity_split=None,
												  min_samples_leaf=1,
												  min_samples_split=2,
												  min_weight_fraction_leaf=0.0,
												  presort=False, random_state=None,
												  splitter='best'), LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
					   intercept_scaling=1,  max_iter=1000,
					   multi_class='warn', n_jobs=None, penalty='l2',
					   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
					   warm_start=False), 
			GaussianNB(), SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False), RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
						   max_depth=4, max_features='auto', max_leaf_nodes=None,
						   min_impurity_decrease=0.0, min_impurity_split=None,
						   min_samples_leaf=1, min_samples_split=2,
						   min_weight_fraction_leaf=0.0, n_estimators=500,
						   n_jobs=None, oob_score=False, random_state=42, verbose=0,
						   warm_start=False),GradientBoostingClassifier(criterion='friedman_mse', init=None,
							   learning_rate=0.1, loss='deviance', max_depth=3,
							   max_features=None, max_leaf_nodes=None,
							   min_impurity_decrease=0.0, min_impurity_split=None,
							   min_samples_leaf=1, min_samples_split=2,
							   min_weight_fraction_leaf=0.0, n_estimators=100,
							   n_iter_no_change=None, presort='auto',
							   random_state=2606, subsample=1.0, tol=0.0001,
							   validation_fraction=0.1, verbose=0,
							   warm_start=False)]

	# loop through algorithms and append the score into the list
	from sklearn.metrics import precision_score
	for i in models:
		model = i
		model.fit(X_train, y_train)
		score = model.score(X_test, y_test)
		y_pred=clf.predict(X_test)
		accuracy.append(score)
		roc.append(roc_auc_score(y_test,y_pred))
		mean_squared_errors.append(precision_score(y_test, y_pred))
		


	# In[47]:


	# create a dataframe from accuracy results
	summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       
	summary


	# In[48]:


	#back propagation nn with 1 hidden layer no of hidden nodes 6 to 20

	kf = KFold(n_splits=10)
	kf.get_n_splits(X)
	clf = GaussianNB()
	print(cross_val_score(clf, X, y, cv=kf, n_jobs=1))
	print(cross_val_score(clf, X, y, cv=kf, n_jobs=1).mean())


	# In[49]:


	from sklearn.neural_network import MLPClassifier

	clf=mlp = MLPClassifier(max_iter=100,hidden_layer_sizes=(17,))
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print(clf)
	print("Accuracy score {}".format(accuracy_score(y_test,y_pred)))
	print("ROC AUC score {}".format(roc_auc_score(y_test,y_pred)))

	model = train_model_info(X_train, y_train, X_test, y_test, MLPClassifier, random_state=2606)


	# In[50]:


	parameters={
	'learning_rate': ["constant", "invscaling", "adaptive"],
	'hidden_layer_sizes': [(6,), (8,), (10,),(12,),(14,),(16,),(18,),(20,)],
	'activation': ["logistic", "relu", "Tanh"],
	'solver': ['sgd', 'adam']
	}
	parameter_space = {
		'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
		'activation': ['tanh', 'relu'],
		'solver': ['sgd', 'adam'],
		'alpha': [0.0001, 0.05],
		'learning_rate': ['constant','adaptive'],
	}
	mlp = MLPClassifier(max_iter=100)

	#grid= GridSearchCV(estimator=MLPClassifier,param_grid=parameters,n_jobs=-1,verbose=2,cv=10)
	clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
	grid.fit(X_train,y_train)

	#print(grid.best_params_) 

	#print(grid.best_estimator_)

	#pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'mean_fit_time','params']]


	# In[51]:


	from sklearn.neural_network import MLPClassifier
	for i in range(6,20):
		kf = KFold(n_splits=10)
		kf.get_n_splits(X)
		clf=mlp = MLPClassifier(max_iter=100, hidden_layer_sizes=(i))
		clf.fit(X_train,y_train)
		y_pred=clf.predict(X_test)
		#print(cross_val_score(clf, X, y, cv=kf, n_jobs=1))
		#print(cross_val_score(clf, X, y, cv=kf, n_jobs=1).mean())
		#print(clf)
		print("Accuracy score {"+str(i)+"}:"+str(accuracy_score(y_test,y_pred)))
		print("ROC AUC score {"+str(i)+"}:"+str(roc_auc_score(y_test,y_pred)))


	# In[52]:



	kf = KFold(n_splits=10)
	kf.get_n_splits(X)
	clf=mlp = MLPClassifier(activation='logistic', alpha=0.00005, batch_size='auto', beta_1=0.9,
				  beta_2=0.999, early_stopping=False, epsilon=1e-08,
				  hidden_layer_sizes=17, learning_rate='adaptive',
				  learning_rate_init=0.01, max_iter=100, momentum=0.9,
				  n_iter_no_change=100, nesterovs_momentum=True, power_t=0.5,
				  random_state=2000, shuffle=True, solver='adam', tol=0.001,
				  validation_fraction=0.1, verbose=False, warm_start=False)
	print(cross_val_score(clf, X, y, cv=kf, n_jobs=1))
	print(cross_val_score(clf, X, y, cv=kf, n_jobs=1).mean())
	# In[ ]:



@app.route('/')
def index():
	return render_template('index2.html')

# @app.route('/api/v01', methods=['POST'])
# def predict():
    # # get data from post request
    # data = request.get_json(force=True)

    # # Load the model
    # model = pickle.load(open('model.pkl', 'rb'))

    # # storing data
    # age = data['age']
    # sex = data['sex']
    # cp = data['cp']
    # trestbps = data['trestbps']
    # chol = data['chol']
    # fbs = data['fbs']
    # restecg = data['restecg']
    # thalach = data['thalach']
    # exang = data['exang']
    # oldpeak = data['oldpeak']
    # slope = data['slope']
    # ca = data['ca']
    # thal = data['thal']

    # # Making array
    # X = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang , oldpeak, slope, ca, thal]]

    # # convert it to array and make prediction
    # prediction = model.predict(X)

    # # Take the first value of prediction
    # output = prediction[0]
    
    # # Changing to string
    # if output == 0:
        # output = 'No. Model predicts that patient got no heart disease'
    # else:
        # output = 'Yes. Model predicts that patient got heart disease'

    # # return as a json
    # return jsonify(output)

# # Code to run server

app.secret_key = 'any random string'

if __name__ == '__main__':
    app.run(port=1000, debug=True)

