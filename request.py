# -*- coding: utf-8 -*-
'''
This file request server in json 
'''

import requests

url = 'http://127.0.0.1:1000/api/v01'
#63,1,3,145,233,1,0,150,0,2.3,0,0,1
r = requests.post(url, json={'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 'fbs' :1, 'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak':2.3, 'slope': 0, 'ca': 0, 'thal': 1 })

# 68,1,0,144,193,1,1,141,0,3.4,1,2,3,0
#print("Target Actual value is 0/No Heart Disease")

#63,1,3,145,233,1,0,150,0,2.3,0,0,1
print("Target Actual value is 1/Yes Heart Disease")

print(r.json())