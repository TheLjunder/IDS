import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Dohvaćanje putanje mape u kojoj se nalazi trenutna datoteka.
from pathlib import Path
currentPath = Path(__file__).parent.absolute()

# Postavljanje putanje za skup podataka kojim će se trenirati model za detekciju.
trainingDataPath = Path.joinpath(currentPath, "KDD Cup 1999 Data/kddcup.data_10_percent.gz")

# Postavljanje putanje za skup podataka kojim će se evaluirati uspješnost napravljenog modela za detekciju.
# TODO

# Uvoz skupa podataka #

# Prema dokumentaciji skupa podataka dodajemo nazive stupaca
# te pridodajemo dodatan naziv stupca koji označava naziv napada.
columnHeaderNames =['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
                    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 
                    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_name',]

# Prema dokumentaciji skupa podataka dodjemo nazive napada
# zajedno s njihovim kategorijama. Postoje 4 glavne kategorije.
attacksNamesAndCategory = {
'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}

# Učitavanje početnog skupa podataka za treniranje.
trainDF = pd.read_csv(trainingDataPath, names = columnHeaderNames)

# Učitavanje početnog skupa podataka za testiranje.

# Dodavanje stupca koji će označavati kategoriju napada u skup podataka za treniranje.
trainDF['attack_category'] = trainDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])

trainDF.shape

# traindDF = df.dropna(axis='columns')# drop columns with NaN
  
# df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

# corr = df.apply(lambda x: x.factorize()[0]).corr()
  
# plt.figure(figsize =(15, 12))
  
# sns.heatmap(corr)
  
# plt.show()

# df.drop('num_root',axis = 1,inplace = True)

# df.drop('srv_serror_rate',axis = 1,inplace = True)

# df.drop('srv_rerror_rate',axis = 1, inplace=True)

# df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

# df.drop('dst_host_serror_rate',axis = 1, inplace=True)

# df.drop('dst_host_rerror_rate',axis = 1, inplace=True)

# df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

# df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)

print(trainDF.head())

print(trainDF.columns)

# Pretvorimo simboličke pokazatelje u numeričke.
pmap = {'icmp':0,'tcp':1,'udp':2}
trainDF['protocol_type'] = trainDF['protocol_type'].map(pmap)

fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
trainDF['flag'] = trainDF['flag'].map(fmap)

trainDF.drop('service',axis = 1,inplace= True)

trainDF.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

trainDF = trainDF.drop(['attack_name',], axis=1)
print(trainDF.shape)

# Target variable and train set
Y = trainDF[['attack_category']]
X = trainDF.drop(['attack_category',], axis=1)

sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split test and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

from sklearn.naive_bayes import GaussianNB

model1 = GaussianNB()

start_time = time.time()
model1.fit(X_train, Y_train.values.ravel())
end_time = time.time()

print("Training time: ",end_time-start_time)

start_time = time.time()
Y_test_pred1 = model1.predict(X_test)
end_time = time.time()

print("Testing time: ",end_time-start_time)

print("Train score is:", model1.score(X_train, Y_train))
print("Test score is:",model1.score(X_test,Y_test))