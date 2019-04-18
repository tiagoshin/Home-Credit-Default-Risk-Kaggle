import numpy as np
import pandas as pd
import logging
import io
import os
import sys

os.chdir('/dados/home-credit')


logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('agg_hcg')

#FIX ME
train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')
br = pd.read_csv('bureau.csv')

logging.info("Read_csv concluido")

#Declaração de lista de funções a serem consideradas
functions = ["count", "first", "last", "sum", "mean", "min", "max", "std"]
functions_teste = ["count", "sum"]

#mad demora

#Declaração de funções a serem consideradas por tipo de variavel
d_cat_func = {"object": ["count", "first", "last", "mode"],
				"float64": ["count", "first", "last", "sum", "mean", "mad", "min", "max", "std"],
				"int64": ["count", "first", "last", "mode", "sum", "min", "max"]}

#Variaveis a serem dropadas da função de agregação
drop_bureau = ["SK_ID_BUREAU"]


def agg_table(maindf, 
				aggdf, 
				drop_list = None, 
				functions = functions):

	""""
    Agrega dados de uma tabela secundaria na tabela principal de acordo com algumas funcoes e colunas determiandas.
    
	INPUT
	maindf: dataframe principal que ficara a esquerda na agregacao
	aggdf: dataframe a ser agregado pela direita no maindf
	drop_list: lista de colunas a ser dropadas do aggdf
	functions: funcoes a serem consideradas na agregacao
    
	OUTPUT
	maindf: tabela resultado da agregcao de aggdf a maindf de acordo com as funcoes definidas
	""" 

	#Dropar colunas do df a ser agregado
	aggdf = aggdf.drop(drop_list, axis = 1)
	
	#Criação do groupby geral
	gb_df = aggdf.groupby("SK_ID_CURR")
	
	#Set index da variavel que linka dfs
	maindf = maindf.set_index("SK_ID_CURR")

	logging.info("DFs prontos para calculo de gb e agregacao")

	for func in functions: 
		
		#Cria lista de colunas a serem consideradas de acordo com função
		col = []
		for coluna in list(aggdf.columns)[1:]:
			if func in d_cat_func[str(aggdf[coluna].dtype)]:
				col.append(coluna)
		
		#Calculo gb da função 
		df_gb_func = gb_df[col].agg(func)
		
		
		#Inclusão da função no nome da variavel
		for coluna in list(df_gb_func.columns):
			df_gb_func = df_gb_func.rename(columns = {coluna: "{}{}{}".format(coluna,"_", func)})

		logging.info("DF da funcao {} criado".format(func))
		
		#Join df principal e agg
		maindf = maindf.join(df_gb_func)

		logging.info("DF da funcao {} agregado".format(func))
		
	return maindf
	
	
testfinal = agg_table(maindf = test, 
				aggdf = br, 
				drop_list = drop_bureau, 
				functions = functions)

trainfinal = agg_table(maindf = train, 
				aggdf = br, 
				drop_list = drop_bureau, 
				functions = functions)

dic = {"train": trainfinal,"teste": testfinal}

np.save("dic_agg_burau.npy", dic)