# Home Credit Default Risk - Competition on Kaggle
source: https://www.kaggle.com/c/home-credit-default-risk
data structure: https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png

The goal is to predict the default risk for a given loan, using features from 7 data sources (see data structure link above) including current application information, bureau data and previous loans made by Home Credit Group.
For this approach, we used only current application and bureau data. Though we could improve performance if all data were explored.

0 - EDA (Exploratory Data Analysis)
1 - agg_bureau - feature generation by aggregation
2 - tratamento - perform encoding and cleaning of missing values 
3 - hcg_model - apply random forest classifier
4 - hcg_submission - submit response to kaggle

Made in June 2018 by Tiago Shin & Gabriela Gonzalez