# Apache Spark es compatible con Python 3.5
# no es compatible con versiones superiores de Python 3.6, 3.7 y 3.8
# por tanto revisar librerias para verificar si pueden correr en Apache Spark 
# los scrips estan escrito para ejecutarse en Apache Spark Data warehouse
# 
# Apache Spark version 3.0.0-preview2 para windows
#
# maoherr@gmail.com

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("ClientsBank")
sc = SparkContext(conf = conf)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels
from scipy import stats
# from pingouin import pairwise_ttests # libreria compatible con python 3.6 y 3.7

# Origen del Dataset
# https://www.kaggle.com/skverma875/bank-marketing-dataset
# Dataset: "bank-full.csv"

# histograma de balance
bank = pd.read_csv("file:///CSpark/bank-full.csv")
plt.hist(bank['balance'],bins = 30);
plt.show()

# histogramas de variables
plt.figure(figsize=(15,10))
vars_to_look = ['marital','education','default','job','housing','loan']
for i, var in enumerate(vars_to_look):
    plt.subplot(2,3,i+1)
    if i ==3:
        plt.xticks(rotation = 90)
    sns.countplot(bank[var])
    plt.title("Count plot of " + var)
plt.show()

# Saldo bancario promedio de los clientes 
print("")
print("el promedio del balance de los clientes es: "+ str(bank['balance'].mean()))
print("")

# prueba estadistica de valor nulo
print("prueba de valor nulo H0=mean")
print(stats.ttest_1samp(bank['balance'], popmean=1341.122))
print("")

# grafico balance bancario entre los que pidieron o no prestamos
ax = sns.stripplot(x="loan", y="balance", data=bank)
plt.ylabel('balance')
plt.show()

# diferencias entre los que pidieron o no prestamos
print("descripción de no prestamistas")
print(bank[bank.loan=="no"].balance.describe())
print("descripción de prestamistas")
print(bank[bank.loan=="yes"].balance.describe())
print("")

statistic, pvalue = stats.ttest_ind(bank[bank.loan=="yes"].balance, bank[bank.loan=="no"].balance, equal_var=False)
print("el valor de statistic es: " + str(statistic))
print("el valor de pvalue es: " + str(pvalue))
print("")

# diferencias por educación
# categorias de educación
print ("las categorias de educación son: ")
print(bank["education"].unique())

# grafico balance bancario en al categoria education
ax = sns.stripplot(x="education", y="balance", data=bank)
plt.ylabel('balance')
plt.show()

# test ANOVA 
mod = ols('balance ~ education', data=bank).fit()  
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print("")

print("descripción de primary")
print(bank[bank.education=="primary"].balance.describe())
print("descripción de secondary")
print(bank[bank.education=="secondary"].balance.describe())
print("descripción de tertiary")
print(bank[bank.education=="tertiary"].balance.describe())
print("descripción de unknown")
print(bank[bank.education=="unknown"].balance.describe())
print("")

# Grafico de barras educación
mean_balance_education=bank.groupby(by="education").balance.mean()
sns.barplot(x=mean_balance_education.index, y=mean_balance_education.values, color="green")
plt.ylabel("Mean balance")
plt.show()

# test por pares - pairwise 𝑡- test.
gb = bank.groupby(['education'])
group_names = bank["education"].unique()
print("\t\t\tstatistic\t\tpvalue")
for i in range(len(group_names)):
  for j in range(i+1, len(group_names)):
    group1 = gb[["balance"]].get_group(group_names[i])
    group2 = gb[["balance"]].get_group(group_names[j])
    stat, pvalue = stats.ttest_ind(group1, group2, equal_var = False)
    print(group_names[i] + " vs. " + group_names[j] + "\t" + str(stat[0]) + "\t" + str(pvalue[0]))

boxplot = bank.boxplot(column=['balance'], by="job",figsize=(14,4))
boxplot.axes.set_title("")
plt.ylabel('balance')
plt.show()

plt.figure(figsize=(14,4))
ax = sns.stripplot(x="job", y="balance", data=bank)
plt.ylabel('balance')
plt.show()

mod = ols('balance ~ job', data=bank).fit()  
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print("")



