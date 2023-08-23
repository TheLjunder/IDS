# Uvoz potrebnih biblioteka
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xlsxwriter
from colorama import Fore, Style
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xlsxwriter import workbook, worksheet

#Globalne varijable
AttackCategoryList = []
StatisticsDF = pd.DataFrame()
AttacksPerCategoryDF = pd.DataFrame()
writer = pd.ExcelWriter('IDS.xlsx', engine = 'xlsxwriter', engine_kwargs={'options':{'strings_to_formulas': False}})

# Početak programa
print("=" * 30)
print(Fore.GREEN + "| Sustav za oktrivanje upada |" + Style.RESET_ALL)
print("=" * 30)
print(Fore.YELLOW + "Učitavam program." + Style.RESET_ALL)

# Dohvaćanje putanje mape u kojoj se nalazi trenutna datoteka.
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
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_name']

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



# Izrada funkcije koja će raditi s modelima 
def useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, model, modelName):
    print(Fore.YELLOW + "Započinjem obradu " + modelName + " modela." + Style.RESET_ALL)
    # Treniranje modela
    print(Fore.YELLOW + "Započinjem treniranje modela." + Style.RESET_ALL)
    startTime = time.time()
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    endTime = time.time()
    trainingTime = endTime-startTime
    print(Fore.YELLOW + "Treniranje modela završeno." + Style.RESET_ALL)
    
    # Testiranje modela
    print(Fore.YELLOW + "Započinjem testiranje modela." + Style.RESET_ALL)
    startTime = time.time()
    prediction = model.predict(testDF_X)
    endTime = time.time()
    testingTime = endTime-startTime
    print(Fore.YELLOW + "Testiranje modela završeno." + Style.RESET_ALL)

    # Izračunavanje statistike
    print(Fore.YELLOW + "Izračunavam statističke podatke modela." + Style.RESET_ALL)
    confusionMatrix = confusion_matrix(testDF_Y, prediction)
    truePositive = np.diag(confusionMatrix)
    falsePositive = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
    falseNegative = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
    trueNegative = confusionMatrix.sum() - (falsePositive + falseNegative + truePositive)
    precision = precision_score(testDF_Y, prediction, average = 'weighted')
    accuracy = accuracy_score(testDF_Y, prediction)

    # Ispis statistike
    print()
    print(Fore.GREEN + "Statistika modela " + modelName + Style.RESET_ALL) 
    print("Vrijeme treniranja: ",trainingTime)
    print("Vrijeme testiranja: ",testingTime)
    print('True Positive po kategoriji: ' + str(truePositive))
    print('False Positive po kategoriji: ' + str(falsePositive))
    print('False Negative po kategoriji: ' + str(falseNegative))
    print('True Negative po kategoriji: ' + str(trueNegative))
    print("Preciznost modela: " + str(precision))
    print("Točnost modela: " + str(accuracy))
    print()

    print(Fore.YELLOW + "Prirpemam statističke podatke za ispis u MS Excel datoteku." + Style.RESET_ALL)
    # Dodavanje općih pokazatelja uspješnosti modela u statistički skup podataka.
    indexStatisticalData = ['Vrijeme treniranja', 'Vrijeme testiranja', 'Preciznost', 'Tocnost']
    StatisticsDF = pd.DataFrame(index = indexStatisticalData)

    # Dodavanje podataka u statistički skup podataka.
    StatisticsDF[modelName] = [trainingTime, testingTime, precision, accuracy]

    # Priprema skupa podataka koji će se koristiti u statističke svrhe a, 
    # sadržati će predviđanja po kategorijama napada.
    indexAttackPerCategory = [AttackCategoryList[0], AttackCategoryList[1], AttackCategoryList[2], AttackCategoryList[3], AttackCategoryList[4]]
    AttacksPerCategoryDF = pd.DataFrame(index = indexAttackPerCategory)

    # Dodavanje podataka u skup podataka s kategorijama.
    AttacksPerCategoryDF['TP'] = truePositive
    AttacksPerCategoryDF['FP'] = falsePositive
    AttacksPerCategoryDF['FN'] = falseNegative
    AttacksPerCategoryDF['TN'] = trueNegative

    print(Fore.YELLOW + "Spremanje u MS Excel datoteku." + Style.RESET_ALL)
    # Spremanje statistike u datoteku programa MS Excel.
    workbook = writer.book
    worksheet = workbook.add_worksheet(modelName)
    worksheet.set_column(0, 0, 18)
    worksheet.set_column(1, 1, 20)
    worksheet.set_column(4, 7, 10)
    writer.sheets[modelName] = worksheet
    StatisticsDF.to_excel(writer, sheet_name = modelName, startrow = 0, startcol = 0)
    AttacksPerCategoryDF.to_excel(writer, sheet_name = modelName, startrow = 0, startcol = 2)

    print(Fore.YELLOW + "Obrada završena." + Style.RESET_ALL)

# Učitavanje početnog skupa podataka za treniranje modela.
print(Fore.YELLOW + "Učitavam skup podataka za treniranje." + Style.RESET_ALL)
trainDF = pd.read_csv(trainingDataPath, names = columnHeaderNames)

# Učitavanje početnog skupa podataka za testiranje modela.
print(Fore.YELLOW + "Učitavam skup podataka za testiranje." + Style.RESET_ALL)
testDF = pd.read_csv(testingDataPath, names = columnHeaderNames)

# Dodavanje stupca koji će označavati kategoriju napada u skupu podataka.
print(Fore.YELLOW + "Dodajem kategorije napada u skupove podataka." + Style.RESET_ALL)
trainDF['attack_category'] = trainDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])
testDF['attack_category'] = testDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])

# Dohvaćanje indeksnih oznaka kategorija napada
countAttackCategoryTestDF = testDF['attack_category'].value_counts()
AttackCategoryList = countAttackCategoryTestDF.index.tolist()

# Pretvaranje simboličkih pokazatelja u numeričke.
print(Fore.YELLOW + "Pretvaram simboličke pokazatelje u numeričke." + Style.RESET_ALL)
pmap = {'icmp':0,'tcp':1,'udp':2}
trainDF['protocol_type'] = trainDF['protocol_type'].map(pmap)
testDF['protocol_type'] = testDF['protocol_type'].map(pmap)
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
trainDF['flag'] = trainDF['flag'].map(fmap)
testDF['flag'] = testDF['flag'].map(fmap)

# Uklonimo simboličke pokazatelje iz skupa podataka 
print(Fore.YELLOW + "Oblikujem skupove podataka za unos u model." + Style.RESET_ALL)
print(Fore.YELLOW + "Uklanjam \"Service\" simbolički pokazatelj." + Style.RESET_ALL)
trainDF.drop('service',axis = 1,inplace= True)
testDF.drop('service',axis = 1,inplace= True)
print(Fore.YELLOW + "Uklanjam \"Attack name\" simbolički pokazatelj." + Style.RESET_ALL)
trainDF = trainDF.drop(['attack_name',], axis=1)
testDF = testDF.drop(['attack_name',], axis=1)

# Postavimo varijable za treniranje i testiranje
print(Fore.YELLOW + "Postavljam X i Y varijable za unos skupa podataka u model." + Style.RESET_ALL)
trainDF_Y = trainDF[['attack_category']]
trainDF_X = trainDF.drop(['attack_category',], axis=1)
testDF_Y = testDF[['attack_category']]
testDF_X = testDF.drop(['attack_category',], axis=1)

# Normaliziramo skup podataka po MinMax skali kako bi 
# model bolje funkcionirao
print(Fore.YELLOW + "Normaliziram skupove podataka u X i Y varijablama." + Style.RESET_ALL)
scaler = MinMaxScaler()
trainDF_X = scaler.fit_transform(trainDF_X)
testDF_X = scaler.fit_transform(testDF_X)

# Napravimo instance modela za različite
# metode strojnog učenja
print(Fore.YELLOW + "Kreiram instancu modela za Gaussian Naive Bayes metodu." + Style.RESET_ALL)
modelGNB = GaussianNB()
modelDT = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
modelRF = RandomForestClassifier(n_estimators = 30)
modelSVC = SVC(gamma = 'scale')
modelLR = LogisticRegression(max_iter = 1200000)
modelGB = GradientBoostingClassifier(random_state = 0)

# Unos parametara u pojednini model pomoću
# metode koja ujedno vraća i statističke podatke.
# useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelGNB, "Gaussian Naive Bayes")
# useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelDT, "Decission Tree")
# useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelRF, "Random Forest")
# useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelSVC, "Support Vector Classifier")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelLR, "Logistic Regression")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelGB, "Gradient Boosting")

# Spremanje konačne MS Excel datoteke.
writer.close()