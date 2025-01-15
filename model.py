import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_df=pd.read_csv("train_u6lujuX_CVtuZ9i.csv") #Training data
test_df=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv") #Testing data
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)



features=train_df.drop(["Loan_Status", "Loan_ID"], axis=1) #Input values #Loan ID is not needed to train the model
labels=train_df["Loan_Status"].copy() #Targeted feature

features_test=test_df.drop(["Loan_Status", "Loan_ID"], axis=1)
labels_test=test_df["Loan_Status"].copy()

# Making categorical values into numerical

features_cat=features.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"], axis=1) #Only categorical data
features_num=features[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]].copy() #Only numerical data

# Filling the null values with mean value

imputer= SimpleImputer(strategy="mean")
imputer.fit(features_num)
X=imputer.transform(features_num)
features_num1=pd.DataFrame(X, columns=features_num.columns, index=features_num.index)

encoder=OneHotEncoder()
features_cat_encoded=encoder.fit_transform(features_cat)
encode_col=encoder.get_feature_names_out(features_cat.columns)
features_num_encoded=pd.DataFrame(features_cat_encoded.toarray(), columns=encode_col)

labels_encoded = encoder.fit_transform(labels.values.reshape(-1, 1)).toarray()
labels_test_encoded = encoder.transform(labels_test.values.reshape(-1, 1)).toarray()

num_attr=list(features_num)
cat_attr=list(features_cat)
num_pipe=Pipeline([("imputer", SimpleImputer(strategy="mean")), ("std_scl", StandardScaler())])
full_pipeline=ColumnTransformer([("num", num_pipe, num_attr), ("cat", OneHotEncoder(), cat_attr)])

train_prepared=full_pipeline.fit_transform(features) # preparing the entire data with the pipeline
test_prepared=full_pipeline.transform(features_test) # using the same pipeline for the test data

# @@@@@@@@@@@@@@ MODEL TRAINING @@@@@@@@@@@@@@@@@@@@@

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



svc=SVC()
svc.fit(train_prepared, labels)
predictions=svc.predict(test_prepared)
## ACCURACY = 0.9

import joblib

joblib.dump(full_pipeline, "pipeline.pkl")
joblib.dump(svc, "model.pkl")

