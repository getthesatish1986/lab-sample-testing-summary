# -*- coding: utf-8 -*-
"""
Created on Fri May 20 07:44:07 2022

@author: Satish S
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report


labtest = pd.read_excel(r"E:\D Drive\College work\subject\Skill course training\360 digiTMG\Project\Project 2\sample_data_final.xlsx")

labtest.columns
labtest.dtypes
labtest.isna().sum()
dup = labtest.duplicated()
sum(dup)

df = labtest.drop(['Patient_ID','Test_Booking_Date','Sample_Collection_Date','Agent_ID'],axis=1)

df_new_1 = pd.get_dummies(df.iloc[:, :-1], drop_first = True)
df_new_1 = df_new_1.join(df['Reached_On_Time'])

labelencoder = LabelEncoder()
df_new_1['Reached_On_Time']= labelencoder.fit_transform(df_new_1['Reached_On_Time'])


plt.boxplot(df_new_1['Patient_Age'])
plt.boxplot(df_new_1['Test_Booking_Time_HH_MM'])
plt.boxplot(df_new_1['Scheduled_Sample_Collection_Time_HH_MM'])
plt.boxplot(df_new_1['Cut-off time_HH_MM'])
plt.boxplot(df_new_1['Agent_Location_KM'])
plt.boxplot(df_new_1['Time_Taken_To_Reach_Patient_MM'])
plt.boxplot(df_new_1['Time_For_Sample_Collection_MM'])
plt.boxplot(df_new_1['Lab_Location_KM'])
plt.boxplot(df_new_1['Time_Taken_To_Reach_Lab_MM'])

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Patient_Age'])

df_new_1['Patient_Age'] = winsor.fit_transform(df[['Patient_Age']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Test_Booking_Time_HH_MM'])

df_new_1['Test_Booking_Time_HH_MM'] = winsor.fit_transform(df[['Test_Booking_Time_HH_MM']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Agent_Location_KM'])

df_new_1['Agent_Location_KM'] = winsor.fit_transform(df[['Agent_Location_KM']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Time_Taken_To_Reach_Patient_MM'])

df_new_1['Time_Taken_To_Reach_Patient_MM'] = winsor.fit_transform(df[['Time_Taken_To_Reach_Patient_MM']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Time_For_Sample_Collection_MM'])

df_new_1['Time_For_Sample_Collection_MM'] = winsor.fit_transform(df[['Time_For_Sample_Collection_MM']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Lab_Location_KM'])

df_new_1['Lab_Location_KM'] = winsor.fit_transform(df[['Lab_Location_KM']])

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Cut-off time_HH_MM'])

df_new_1['Cut-off time_HH_MM'] = winsor.fit_transform(df[['Cut-off time_HH_MM']])

df_new_1.columns = ['Patient_Age', 'Test_Booking_Time_HH_MM',
       'Scheduled_Sample_Collection_Time_HH_MM', 'Cut_off_time_HH_MM',
       'Agent_Location_KM', 'Time_Taken_To_Reach_Patient_MM',
       'Time_For_Sample_Collection_MM', 'Lab_Location_KM',
       'Time_Taken_To_Reach_Lab_MM', 'Patient_Gender_Male', 'Test_Name_CBC',
       'Test_Name_Complete_Urinalysis', 'Test_Name_Fasting_blood_sugar',
       'Test_Name_H1N1', 'Test_Name_HbA1c', 'Test_Name_Lipid_Profile',
       'Test_Name_RTPCR', 'Test_Name_TSH', 'Test_Name_Vitamin_D_25Hydroxy',
       'Sample_Swab', 'Sample_Urine', 'Sample_blood',
       'Way_Of_Storage_Of_Sample_Normal', 'Cut_off_Schedule_Sample_by_5pm',
       'Traffic_Conditions_Low_Traffic', 'Traffic_Conditions_Medium_Traffic',
       'Reached_On_Time']

df_new_1.columns

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Reached_On_Time ~ Test_Booking_Time_HH_MM + Scheduled_Sample_Collection_Time_HH_MM + Cut_off_time_HH_MM + Agent_Location_KM + Time_Taken_To_Reach_Patient_MM + Time_For_Sample_Collection_MM + Lab_Location_KM +Time_Taken_To_Reach_Lab_MM + Patient_Gender_Male + Test_Name_CBC + Test_Name_Complete_Urinalysis + Test_Name_Fasting_blood_sugar +Test_Name_H1N1+Test_Name_HbA1c+Test_Name_Lipid_Profile+Test_Name_RTPCR+Test_Name_TSH+Test_Name_Vitamin_D_25Hydroxy+Sample_Swab+ Sample_Urine+ Sample_blood+Way_Of_Storage_Of_Sample_Normal+Cut_off_Schedule_Sample_by_5pm+Traffic_Conditions_Low_Traffic+Traffic_Conditions_Medium_Traffic', data = df_new_1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(df_new_1.iloc[ :, :-1 ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df_new_1.Reached_On_Time, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
df_new_1["pred"] = np.zeros(1019)
# taking threshold value and above the prob value will be treated as correct value 
df_new_1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(df_new_1["pred"], df_new_1["Reached_On_Time"])
classification

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df_new_1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Reached_On_Time ~ Test_Booking_Time_HH_MM + Scheduled_Sample_Collection_Time_HH_MM + Cut_off_time_HH_MM + Agent_Location_KM + Time_Taken_To_Reach_Patient_MM + Time_For_Sample_Collection_MM + Lab_Location_KM +Time_Taken_To_Reach_Lab_MM + Patient_Gender_Male + Test_Name_CBC + Test_Name_Complete_Urinalysis + Test_Name_Fasting_blood_sugar +Test_Name_H1N1+Test_Name_HbA1c+Test_Name_Lipid_Profile+Test_Name_RTPCR+Test_Name_TSH+Test_Name_Vitamin_D_25Hydroxy+Sample_Swab+ Sample_Urine+ Sample_blood+Way_Of_Storage_Of_Sample_Normal+Cut_off_Schedule_Sample_by_5pm+Traffic_Conditions_Low_Traffic+Traffic_Conditions_Medium_Traffic', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(306)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Reached_On_Time'])
confusion_matrix

accuracy_test = (57 + 237)/(306) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Reached_On_Time"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Reached_On_Time"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :-2 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(713)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Reached_On_Time'])
confusion_matrx

accuracy_train = (138 + 565)/(713)
print(accuracy_train)

import pickle
file_name=r'E:\D Drive\College work\subject\Skill course training\360 digiTMG\Project\Project 2\my_file.pkl'
f = open(file_name,'wb')
pickle.dump(model,f)
f.close()

#Test_Booking_Time_HH_MM + Scheduled_Sample_Collection_Time_HH_MM + Cut_off_time_HH_MM + Agent_Location_KM + 
#Time_Taken_To_Reach_Patient_MM + Time_For_Sample_Collection_MM + Lab_Location_KM +Time_Taken_To_Reach_Lab_MM + 
#Patient_Gender_Male + Test_Name_CBC + Test_Name_Complete_Urinalysis + Test_Name_Fasting_blood_sugar +Test_Name_H1N1+Test_Name_HbA1c+
#Test_Name_Lipid_Profile+Test_Name_RTPCR+Test_Name_TSH+Test_Name_Vitamin_D_25Hydroxy+Sample_Swab+ Sample_Urine+ 
#Sample_blood+Way_Of_Storage_Of_Sample_Normal+Cut_off_Schedule_Sample_by_5pm+Traffic_Conditions_Low_Traffic+
#Traffic_Conditions_Medium_Traffic'

model = pickle.load(open(file_name,'rb'))
a = pd.DataFrame([12.4,13,17,7,14,10,13,26,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0])
fi = a.transpose()
fi.columns = ['Test_Booking_Time_HH_MM' , 'Scheduled_Sample_Collection_Time_HH_MM' , 'Cut_off_time_HH_MM' , 'Agent_Location_KM' , 
              'Time_Taken_To_Reach_Patient_MM' , 'Time_For_Sample_Collection_MM' , 'Lab_Location_KM' ,'Time_Taken_To_Reach_Lab_MM' ,
              'Patient_Gender_Male' , 'Test_Name_CBC' , 'Test_Name_Complete_Urinalysis' , 'Test_Name_Fasting_blood_sugar' ,'Test_Name_H1N1','Test_Name_HbA1c',
              'Test_Name_Lipid_Profile','Test_Name_RTPCR','Test_Name_TSH','Test_Name_Vitamin_D_25Hydroxy','Sample_Swab','Sample_Urine',
              'Sample_blood','Way_Of_Storage_Of_Sample_Normal','Cut_off_Schedule_Sample_by_5pm','Traffic_Conditions_Low_Traffic',
              'Traffic_Conditions_Medium_Traffic']
print(model.predict(fi))
