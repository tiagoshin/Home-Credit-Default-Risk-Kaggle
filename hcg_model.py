import numpy as np
import pandas as pd
import logging
import io
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

os.chdir('/home/gonzalez/gdata/hcg_gabi_shin')

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger('model_hcg')

#carregar bases
dic_dfs = np.load("dic_treat.npy").tolist()
train = dic_dfs["train"]
test = dic_dfs["test"]

def prep_data(train = train):

	#Preparando o train
	X = train.drop("target", axis = 1)
	y = train["target"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
	
	return  X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prep_data()

logging.info("Data prep concluido")

def run_model(model = "forest", X_train = X_train, y_train = y_train):

	if model == "forest":
	    rf = RandomForestClassifier()
	    rf.fit(X_train, y_train)
	    
	return rf

md = run_model()
logging.info("Modelo treinado")

def predict_md(md = md, X = test):
	y_pred = md.predict(X)
	return y_pred
	

def metricas(metrica, md, y_pred, y_test):
	
	"""
	IMPUT
	
	metrical: "acc", "confussion"
	md: modelo
	y_pred: target predict
	y_test: target real
	"""
	
	if metrica == "acc":
		score = accuracy_score(y_test, y_pred)
		return score	
	#if metrica = "confussion"

y = predict_md()
logging.info("Prediction feita")

#Prep para submissão - falta renomear a coluna de TARGET
#y = y.reset_index()
#dic = {"sub": y}

#np.save("y_sub.npy", dic)