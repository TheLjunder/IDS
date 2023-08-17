print("=" * 30)
print("| Sustav za oktrivanje upada |")
print("=" * 30)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from colorama import Fore, Style

print(Fore.YELLOW + "Učitavam program." + Style.RESET_ALL)
# Dohvaćanje putanje mape u kojoj se nalazi trenutna datoteka.
from pathlib import Path
currentPath = Path(__file__).parent.absolute()

# Postavljanje putanje za skup podataka kojim će se trenirati model za detekciju.
trainingDataPath = Path.joinpath(currentPath, "KDD Cup 1999 Data/kddcup.data_10_percent.gz")

# Postavljanje putanje za skup podataka kojim će se evaluirati uspješnost napravljenog modela za detekciju.
testingDataPath = Path.joinpath(currentPath, "KDD Cup 1999 Data/kddcup.data.gz")

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

# Prema dokumentaciji skupa podataka dodajemo nazive napada
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
print(Fore.YELLOW + "Učitavam skup podataka za treniranje." + Style.RESET_ALL)
trainDF = pd.read_csv(trainingDataPath, names = columnHeaderNames)

# Učitavanje početnog skupa podataka za testiranje.
print(Fore.YELLOW + "Učitavam skup podataka za testiranje." + Style.RESET_ALL)
testDF = pd.read_csv(testingDataPath, names = columnHeaderNames)

# Dodavanje stupca koji će označavati kategoriju napada u skup podataka.
print(Fore.YELLOW + "Dodajem kategorije napada u skupove podataka." + Style.RESET_ALL)
trainDF['attack_category'] = trainDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])
testDF['attack_category'] = testDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])
countAttackCategoryTestDF = testDF['attack_category'].value_counts()
print(countAttackCategoryTestDF)
attackCategoryList = countAttackCategoryTestDF.index.tolist()
print(attackCategoryList)

# Pretvorimo simboličke pokazatelje u numeričke.
print(Fore.YELLOW + "Pretvaram simboličke pokazatelje u numeričke." + Style.RESET_ALL)
pmap = {'icmp':0,'tcp':1,'udp':2}
trainDF['protocol_type'] = trainDF['protocol_type'].map(pmap)
testDF['protocol_type'] = testDF['protocol_type'].map(pmap)

fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
trainDF['flag'] = trainDF['flag'].map(fmap)
testDF['flag'] = testDF['flag'].map(fmap)

# Uklonimo nebitan simbolički pokazatelj
print(Fore.YELLOW + "Uklanjam service simbolički pokazatelj." + Style.RESET_ALL)
trainDF.drop('service',axis = 1,inplace= True)
testDF.drop('service',axis = 1,inplace= True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(Fore.YELLOW + "Oblikujem skupove podataka za unos u model." + Style.RESET_ALL)
trainDF = trainDF.drop(['attack_name',], axis=1)
testDF = testDF.drop(['attack_name',], axis=1)

# Target variable and train set
trainDF_Y = trainDF[['attack_category']]
trainDF_X = trainDF.drop(['attack_category',], axis=1)
testDF_Y = testDF[['attack_category']]
testDF_X = testDF.drop(['attack_category',], axis=1)

print(Fore.YELLOW + "Normaliziram skupove podataka." + Style.RESET_ALL)
sc = MinMaxScaler()
trainDF_X = sc.fit_transform(trainDF_X)
testDF_X = sc.fit_transform(testDF_X)

from sklearn.naive_bayes import GaussianNB

model1 = GaussianNB()

print(Fore.YELLOW + "Započinjem treniranje modela." + Style.RESET_ALL)
start_time = time.time()
model1.fit(trainDF_X, trainDF_Y.values.ravel())
end_time = time.time()

print("Training time: ",end_time-start_time)

print(Fore.YELLOW + "Započinjem testiranje modela." + Style.RESET_ALL)
start_time = time.time()
Y_test_pred1 = model1.predict(testDF_X)
end_time = time.time()

print("Testing time: ",end_time-start_time)

# Izračunavanje statistike

from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(testDF_Y, Y_test_pred1)
confusionMatrixDF = pd.DataFrame(confusionMatrix, index = ['normal', 'dos', 'probe', 'r2l', 'u2r'], columns = ['normal', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for Gaussian Naives Bayes Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sns.heatmap(confusionMatrixDF, annot=True, annot_kws={"size": 10}) # font size
plt.show(block=False)

tp = np.diag(confusionMatrix)
tp = pd.Series(tp)

tp = tp.set_axis(attackCategoryList)

diagramDF = pd.concat([countAttackCategoryTestDF, tp], axis = 1)
diagramDF.columns = ['Test', 'Predict']

diagramDF.plot(y = ["Test", "Predict"], kind = "bar")
plt.show()

print("Train score is:", model1.score(trainDF_X, trainDF_Y))
print("Test score is:",model1.score(testDF_X, testDF_Y))