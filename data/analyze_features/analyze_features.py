import os 
from os import listdir
from os.path import isfile, join
import pandas as pd

dfIdTest = pd.read_csv('test_id', sep = '\t', header=None, names=["id"])
df1 = pd.read_csv('test_text.pt', sep = '\t', header=None, names=["portuguese"])
df2 = pd.read_csv('test_text.en', sep = '\t', header=None, names=["english"])
dfIdTrain = pd.read_csv('train_id', sep = '\t', header=None, names=["id"])
df3 = pd.read_csv('train_text.pt', sep = '\t', header=None, names=["portuguese"])
df4 = pd.read_csv('train_text.en', sep = '\t', header=None, names=["english"])
dfIdVal = pd.read_csv('val_id', sep = '\t', header=None, names=["id"])
df5 = pd.read_csv('val_text.pt', sep = '\t', header=None, names=["portuguese"])
df6 = pd.read_csv('val_text.en', sep = '\t', header=None, names=["english"])

index_test = dfIdTest.index
index_train = dfIdTrain.index
index_val = dfIdVal.index

# Pegar todas as features que existem em ./how2/features 
onlyfiles = [f for f in listdir("./../how2/features") if isfile(join("./../how2/features", f))]
features_indices_test, features_indices_test_list, features_indices_train, features_indices_train_list, features_indices_val, features_indices_val_list = [], [], [], [], [], []

for i in range(0,len(onlyfiles)):
  onlyfiles[i] = os.path.splitext(onlyfiles[i])[0] # Pegar o nome dos arquivos sem a extensão .pkl
  condition_test = dfIdTest["id"] == onlyfiles[i]
  condition_train = dfIdTrain["id"] == onlyfiles[i]
  condition_val = dfIdVal["id"] == onlyfiles[i]

  if len(index_test[condition_test]) > 0:
    # Adicionar na lista de índices o índice referindo-se ao nome da feature
    features_indices_test.append(index_test[condition_test].tolist())
    features_indices_test_list.append(features_indices_test[-1][0])

  if len(index_train[condition_train]) > 0:
    features_indices_train.append(index_train[condition_train].tolist())
    features_indices_train_list.append(features_indices_train[-1][0])

  if len(index_val[condition_val]) > 0:
    features_indices_val.append(index_val[condition_val].tolist())
    features_indices_val_list.append(features_indices_val[-1][0])

# Agora tenho uma lista com os indices para os quais existem features
dfIdTest_fim = dfIdTest.iloc[sorted(features_indices_test_list)]
df1_fim = df1.iloc[sorted(features_indices_test_list)]
df2_fim = df2.iloc[sorted(features_indices_test_list)]
dfIdTrain_fim = dfIdTrain.iloc[sorted(features_indices_train_list)]
df3_fim = df3.iloc[sorted(features_indices_train_list)]
df4_fim = df4.iloc[sorted(features_indices_train_list)]
dfIdVal_fim = dfIdVal.iloc[sorted(features_indices_val_list)]
df5_fim = df5.iloc[sorted(features_indices_val_list)]
df6_fim = df6.iloc[sorted(features_indices_val_list)]

dfIdTest_fim.to_csv('test.order', index=False, header=False, columns=["id"]) 
df1_fim.to_csv('test.pt', index=False, header=False, columns=["portuguese"]) 
df2_fim.to_csv('test.en', index=False, header=False, columns=["english"]) 
dfIdTrain_fim.to_csv('train.order', index=False, header=False, columns=["id"]) 
df3_fim.to_csv('train.pt', index=False, header=False, columns=["portuguese"]) 
df4_fim.to_csv('train.en', index=False, header=False, columns=["english"]) 
dfIdVal_fim.to_csv('val.order', index=False, header=False, columns=["id"]) 
df5_fim.to_csv('val.pt', index=False, header=False, columns=["portuguese"]) 
df6_fim.to_csv('val.en', index=False, header=False, columns=["english"]) 
