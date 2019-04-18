import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('treat_hcg')

#Libraries setup
os.chdir('/home/gonzalez/gdata/hcg_gabi_shin')

#carregar bases
dic_dfs = np.load("dic_agg_burau.npy").tolist()
train = dic_dfs["train"]
test = dic_dfs["teste"]

#renomear colunass
train = pd.read_csv('application_train.csv')
train.columns = map(str.lower, train.columns)

logging.info("Colunas renomeadas")

#tratamento
#cat_without_nulls - label encoder
#cat_with_nulls - se tem mais de de 30% nulo, fillna com moda
#cont_without_nulls - standart scaler
#cont_with_nulls - se tem mais de 30% nulo dropa, se nao fillna mean

def tratamento (train):
    
	""""
    Trata os datasets agregados. Trata nulos e usa lógicas diferentes para dados categóricos/discretos e contínuos

	INPUT
	train: dataset a ser tratado
    
	OUTPUT
	train: tabela pronta para rodar o random forest
	""" 
    #Identifica colunas com menos de 30% nulos e binariza
    dp = []
    size = train.count().max()
    
    logging.info("Início analise na")
    for ls in train.columns:
        if train[ls].isnull().sum()/train.count().max() > 0.3:
            dp.append(ls)
            train[dp] = train[dp].notnull().map({'False':0, 'True': 1})
    #train.drop(dp, axis=1, inplace=True)   
    #Identifica dados em float64 que se comportam como discretos
    for col in train.columns:
        if str(train[col].dtype) == 'float64':
            if train[col].value_counts().count() < 50:
                train[col] = train[col].fillna(train[col].mode().iloc[0]).astype('int64')
	#Identifica dados em int64 que se comportam como continuos
        if str(train[col].dtype) == 'int64':
            if train[col].value_counts().count() > 50:
                train[col] = train[col].fillna(train[col].mean()).astype('float64') 
	#Trata dados categoricos        
        if str(train[col].dtype) == 'object':
            train[col] = train[col].fillna(train[col].mode().iloc[0])
            train[col] = LabelEncoder().fit_transform(train[col])
	#Trata dados discretos
        if str(train[col].dtype) == 'int64':
            train[col] = train[col].fillna(train[col].mode().iloc[0])
            train[col] = LabelEncoder().fit_transform(train[col])       
    logging.info("Tratamento 1 concluído")     
    nc = []
    for ct in train.columns:
        if str(train[ct].dtype) == 'float64':
            nc.append(ct)
            train[ct] = train[ct].fillna(train[ct].mean())
#            train[ct] = Normalizer().fit_transform(train[nc])
    logging.info("Tratamento 2 concluído")
    return train

train_treat = tratamento(train=train)
test_treat = tratamento(train=test)

dic_treat = {"train": train_treat,"test": test_treat}

np.save("dic_treat.npy", dic_treat)

#testes
#print(df_treat.columns)
#print(df_treat.info())
#print(df_treat.head())
#print(df_treat.amt_credit.isnull().sum())
#print(df_treat.isnull().any())
