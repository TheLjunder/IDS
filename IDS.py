# 1. Inicijalizacija programa

# Uvoz potrebnih biblioteka
from colorama import Fore, Style
from pathlib import Path
import pandas as pd
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# Globalne varijable
# Varijable vezane uz statističke izračune 
TrainingTimeValues = {}
TestingTimeValues = {}
PrecisionValues = {}
RecallValues = {}
FPRValues = {}
TNRValues = {}
AccuracyValues = {}
FMeassureValues={}
# Varijable vezane uz kreiranje MS Excel datoteke
fileName = 'IDS.xlsx'
writer = pd.ExcelWriter(fileName, engine = 'xlsxwriter', engine_kwargs={'options':{'strings_to_formulas': False}})
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

# Ispis početne poruke
print("=" * 30)
print(Fore.GREEN + "| Sustav za oktrivanje upada |" + Style.RESET_ALL)
print("=" * 30)

    # 7. Rad sa modelima strojnog učenja

    # Funkcija koja izvršava treniranje i testiranje modelima strojnog učenja.
    # Zatim, statistički podaci modela se pospremaju na dva različita "mjesta".
    # Jedan dio služi za spremanje podataka u MS Excel datoteku, a drugi za iscrtavanje grafova. 
def useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, model, modelName):

    # Treniranje modela.
    print(Fore.YELLOW + "Započinjem obradu " + modelName + " modela." + Style.RESET_ALL)
    print(Fore.YELLOW + "Započinjem treniranje modela." + Style.RESET_ALL)
    startTime = time.time()
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    endTime = time.time()
    trainingTime = endTime-startTime
    print(Fore.YELLOW + "Treniranje modela završeno." + Style.RESET_ALL)
    
    # Testiranje modela.
    print(Fore.YELLOW + "Započinjem testiranje modela." + Style.RESET_ALL)
    startTime = time.time()
    prediction = model.predict(testDF_X)
    endTime = time.time()
    testingTime = endTime-startTime
    print(Fore.YELLOW + "Testiranje modela završeno." + Style.RESET_ALL)

    # Izračun statističkih pokazatelja na temelju predviđanja modela.
    print(Fore.YELLOW + "Izračunavam statističke podatke modela." + Style.RESET_ALL)
    confusionMatrix = confusion_matrix(testDF_Y, prediction)
    truePositive = np.diag(confusionMatrix)
    falsePositive = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
    falseNegative = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
    trueNegative = confusionMatrix.sum() - (falsePositive + falseNegative + truePositive)
    precision = precision_score(testDF_Y, prediction, average = 'weighted') * 100
    recall = truePositive / (truePositive + falseNegative) * 100
    falsePositiveRate = falsePositive / (falsePositive + trueNegative) * 100
    trueNegativeRate = trueNegative / (trueNegative + falsePositive) * 100
    accuracy = accuracy_score(testDF_Y, prediction) * 100
    fMeassure = 2 * ((precision * recall) / (precision + recall))

    # 1. Spremanje podataka u MS Excel datoteku

    # Priprema statističkih podataka modela za spremanje u MS Excel datoteku.
    print(Fore.YELLOW + "Prirpemam statističke podatke za ispis u MS Excel datoteku." + Style.RESET_ALL)
    indexStatisticalData = ['Vrijeme treniranja', 'Vrijeme testiranja', 'Preciznost', 'Tocnost']
    StatisticalDataDF = pd.DataFrame(index = indexStatisticalData)
    StatisticalDataDF[modelName] = [trainingTime, testingTime, precision, accuracy]
    StatisticalDataPerCategoryDF = pd.DataFrame(index = AttackCategoryList)
    StatisticalDataPerCategoryDF['TP'] = truePositive
    StatisticalDataPerCategoryDF['FP'] = falsePositive
    StatisticalDataPerCategoryDF['FN'] = falseNegative
    StatisticalDataPerCategoryDF['TN'] = trueNegative
    StatisticalDataPerCategoryDF['FPR'] = falsePositiveRate
    StatisticalDataPerCategoryDF['TNR'] = trueNegativeRate
    StatisticalDataPerCategoryDF['Opoziv'] = recall
    StatisticalDataPerCategoryDF['F-mjera'] = fMeassure
    # Spremanje statističkih podataka modela u MS Excel datoteku.
    print(Fore.YELLOW + "Spremanje u MS Excel datoteku." + Style.RESET_ALL)
    workbook = writer.book
    worksheet = workbook.add_worksheet(modelName)
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 18)
    worksheet.set_column(3, 7, 10)
    worksheet.set_column(8, 11, 12)
    writer.sheets[modelName] = worksheet
    StatisticalDataDF.to_excel(writer, sheet_name = modelName, startrow = 0, startcol = 0)
    StatisticalDataPerCategoryDF.to_excel(writer, sheet_name = modelName, startrow = 0, startcol = 3)

    # 2. Spremanje podataka u globalne varijable programa za kasnije iscrtavanje grafova.

    TrainingTimeValues[modelName] = trainingTime
    TestingTimeValues[modelName] = testingTime
    PrecisionValues[modelName] = precision
    RecallValues[modelName] = recall
    FPRValues[modelName] = falsePositiveRate
    TNRValues[modelName] = trueNegativeRate
    AccuracyValues[modelName] = accuracy
    FMeassureValues[modelName] = fMeassure

    print(Fore.YELLOW + "Obrada modela završena." + Style.RESET_ALL)

# 2. Učitvanje skupova podataka

# Dohvaćanje putanje mape u kojoj se nalazi program.
currentPath = Path(__file__).parent.absolute()
# Postavljanje putanje za skup podataka kojim će se trenirati model za detekciju.
trainingDataPath = Path.joinpath(currentPath, "KDD Cup 1999 Data/kddcup.data_10_percent.gz")
# Postavljanje putanje za skup podataka kojim će se evaluirati uspješnost napravljenog modela za detekciju.
testingDataPath = Path.joinpath(currentPath, "KDD Cup 1999 Data/kddcup.data.gz")
# Učitavanje početnog skupa podataka za treniranje modela.
print(Fore.YELLOW + "Učitavam skup podataka za treniranje." + Style.RESET_ALL)
trainDF = pd.read_csv(trainingDataPath, names = columnHeaderNames)
# Učitavanje početnog skupa podataka za testiranje modela.
print(Fore.YELLOW + "Učitavam skup podataka za testiranje." + Style.RESET_ALL)
testDF = pd.read_csv(testingDataPath, names = columnHeaderNames)

# 3. Oblikovanje skupova podataka

# Dodavanje stupca koji označava atribut napada u skupu podataka.
print(Fore.YELLOW + "Dodajem kategorije napada u skupove podataka." + Style.RESET_ALL)
trainDF['attack_category'] = trainDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])
testDF['attack_category'] = testDF.attack_name.apply(lambda r:attacksNamesAndCategory[r[:-1]])
# Dohvaćanje oznaka kategorije napada iz testnog skupa podataka. Oznake su dohvaćene silazno prema broju 
# pojavljivanja unutar testnog skupa podataka. Lista kategorija se kasnije koristi za oblikovanje statističkih skupova podataka.
AttackCategoryListDF = testDF['attack_category'].value_counts()
AttackCategoryList = AttackCategoryListDF.index.tolist()
# Pretvaranje simboličkih pokazatelja u numeričke.
print(Fore.YELLOW + "Pretvaram simboličke pokazatelje u numeričke." + Style.RESET_ALL)
pmap = {'icmp':0,'tcp':1,'udp':2}
trainDF['protocol_type'] = trainDF['protocol_type'].map(pmap)
testDF['protocol_type'] = testDF['protocol_type'].map(pmap)
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
trainDF['flag'] = trainDF['flag'].map(fmap)
testDF['flag'] = testDF['flag'].map(fmap)
# Uklanjanje nevažnog pokazatelja iz skupova podataka. 
print(Fore.YELLOW + "Oblikujem skupove podataka za unos u model." + Style.RESET_ALL)
print(Fore.YELLOW + "Uklanjam \"Service\" simbolički pokazatelj." + Style.RESET_ALL)
trainDF.drop('service',axis = 1,inplace= True)
testDF.drop('service',axis = 1,inplace= True)
# Uklanjane atributa koji označava vrstu zapisa iz skupova podataka.
print(Fore.YELLOW + "Uklanjam \"Attack name\" simbolički pokazatelj." + Style.RESET_ALL)
trainDF = trainDF.drop(['attack_name',], axis=1)
testDF = testDF.drop(['attack_name',], axis=1)

# 4. Inicijalizacija ulaznih varijabli skupova podataka

# Postavljanje varijabla za treniranje i testiranje koje će koristiti model.
# Varijable sa sufiksom X predstavljaju neoznačeni skup podataka.
# varijable sa sufiksom Y predstavljaju izlazne klasifikacije prema kojima model obrađuje skup podataka X.
print(Fore.YELLOW + "Postavljam X i Y varijable za unos skupa podataka u model." + Style.RESET_ALL)
trainDF_Y = trainDF[['attack_category']]
trainDF_X = trainDF.drop(['attack_category',], axis=1)
testDF_Y = testDF[['attack_category']]
testDF_X = testDF.drop(['attack_category',], axis=1)

# 5. Normalizacija ulaznih varijabli modela

# Normalizacija skupova podataka po MinMax skali kako bi model bolje funkcionirao.
print(Fore.YELLOW + "Normaliziram skupove podataka u X i Y varijablama." + Style.RESET_ALL)
scaler = MinMaxScaler()
trainDF_X = scaler.fit_transform(trainDF_X)
testDF_X = scaler.fit_transform(testDF_X)

# 6. Inicijalizacija modela strojnog učenja

# Inicijalizacija modela za različite metode strojnog učenja.
print(Fore.YELLOW + "Inicijaliziram modele strojnog učenja." + Style.RESET_ALL)
modelGNB = GaussianNB()
modelDT = DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
modelRF = RandomForestClassifier(n_estimators = 30)
modelLR = LogisticRegression(max_iter = 1200000)
modelGB = GradientBoostingClassifier(random_state = 0)

# 7. Rad sa modelima strojnog učenja

# Unos parametara u pojednini model pomoću metode za obradu modela
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelGNB, "Gaussian Naive Bayes")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelDT, "Decission Tree")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelRF, "Random Forest")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelLR, "Logistic Regression")
useModel(trainDF_Y, trainDF_X, testDF_Y, testDF_X, modelGB, "Gradient Boosting")

# Finaliziranje i spremanje konačne MS Excel datoteke.
writer.close()
print(Fore.YELLOW + "Kreirana je MS Excel datoteka pod nazivom \"" + str(fileName) + "\" u korijenskoj datoteci ovog Python programa." + Style.RESET_ALL)

# 8. Iscrtavanje i spremanje grafova pokazatelja uspješnosti pojedinog modela.

print(Fore.YELLOW + "Kreiram grafove pojedinih pokazatelja." + Style.RESET_ALL)
# Vrijeme treniranja modela
trainingTimeValuesDF = pd.DataFrame(TrainingTimeValues, index = ['Vremena'])
trainingTimeValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Usporedba vremena treniranja pojedinog modela')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('treniranje.png',bbox_inches='tight')
# Vrijeme testiranja modela
testingTimeValuesDF = pd.DataFrame(TestingTimeValues, index = ['Vremena'])
testingTimeValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Usporedba vremena testiranja pojedinog modela')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('testiranje.png',bbox_inches='tight')
# Preciznost
precisonValuesDF = pd.DataFrame(PrecisionValues, index = ['Vrijendosti pokazatelja'])
precisonValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Preciznost pojedinog modela')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('precision.png',bbox_inches='tight')
# Opoziv
recallValuesDF = pd.DataFrame(RecallValues, index = AttackCategoryList)
recallValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Opoziv po kategorijama')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('recall.png',bbox_inches='tight')
# FPR
falsePositiveValuesDF = pd.DataFrame(FPRValues, index = AttackCategoryList)
falsePositiveValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Stopa pogrešnih klasifikacija napada po kategorijama')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('fpr.png',bbox_inches='tight')
# TNR
trueNegativeValuesDF = pd.DataFrame(TNRValues, index = AttackCategoryList)
trueNegativeValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Stopa točnih klasifikacija normalnih zapisa po kategorijama')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('tnr.png',bbox_inches='tight')
# Točnost
accuracyValuesDF = pd.DataFrame(AccuracyValues, index = ['Vrijendosti pokazatelja'])
accuracyValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('Točnost pojedinog modela')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('accuracy.png',bbox_inches='tight')
# F-mjera
fMeassureValuesDF = pd.DataFrame(FMeassureValues, index = AttackCategoryList)
fMeassureValuesDF.plot(kind = 'bar')
plt.xticks(rotation=0)
plt.title('F-mjera modela po kategorijama')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig('fMeassure.png',bbox_inches='tight')

print(Fore.YELLOW + "Slike grafova statističkih pokazatelja su spremljene u korijenski direktorij ovog programa." + Style.RESET_ALL)

print(Fore.YELLOW + "Kraj programa." + Style.RESET_ALL)