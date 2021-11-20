#%%
# using BP and White Testing for Heteroscedasticity
# going to use statsmodels for the OLS linear regression

# import dependencies
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
# from statsmodels.compat import lzip
import matplotlib.pyplot as plt
# %%
file = ('data\car models fuel economy.csv')
df = pd.read_csv(file)
df
# %%
model1 = smf.ols(formula = 'Cmpg ~ Eng', data = df).fit()
print(model1.summary())
# %%
resid1 = model1.resid
fitted1 = model1.fittedvalues
#QQ plot of residuals
fig1 = sm.qqplot(resid1,fit=True,line='45')
#fitted v. residuals
fig, ax = plt.subplots()
_ = ax.scatter(fitted1, resid1)
# %%
# Breusch-Pagan Test for Heteroscedasticity
bptest = sms.diagnostic.het_breuschpagan(resid1,model1.model.exog)
# White Test for Heteroscedasticity, including squares and cross-product of exog
white_test = sms.diagnostic.het_white(resid1,model1.model.exog)
# %%
print('Breusch-Pagan Test and White LM Tests for Heteroscedasticity')
df1 = pd.DataFrame({'Test Type':['Breusch-Pagan Test', 'White Test'],
                   'Chi-Sq':[bptest[0], white_test[0]], 'DF':[2, 4],
                   'Prob>Chi-Sq':[bptest[1], white_test[1]]})
print(df1)
# %%
# take inverse of engine in this model
model2 = smf.ols(formula = 'Cmpg ~ Inv_eng', data = df).fit()
print(model2.summary())
# %%
resid2 = model2.resid
fitted2 = model2.fittedvalues
#QQ plot of residuals
fig1 = sm.qqplot(resid2,fit=True,line='45')
#fitted v. residuals
fig, ax = plt.subplots()
_ = ax.scatter(fitted2, resid2)
# %%
# Breusch-Pagan Test for Heteroscedasticity
bptest = sms.diagnostic.het_breuschpagan(resid2,model2.model.exog)
# White Test for Heteroscedasticity, including squares and cross-product of exog
white_test = sms.diagnostic.het_white(resid2,model2.model.exog)
# %%
print('Breusch-Pagan Test and White LM Tests for Heteroscedasticity')
df2 = pd.DataFrame({'Test Type':['Breusch-Pagan Test', 'White Test'],
                   'Chi-Sq':[bptest[0], white_test[0]], 'DF':[2, 4],
                   'Prob>Chi-Sq':[bptest[1], white_test[1]]})
print(df2)
# %%
# create new column
df["Inv_cmpg"] = 1/df["Cmpg"] 
df
# %%
# create third model with both predictors transformed
model3 = smf.ols(formula = 'Inv_cmpg ~ Inv_eng', data = df).fit()
print(model3.summary())
# %%
resid3 = model3.resid
fitted3 = model3.fittedvalues
#QQ plot of residuals
fig1 = sm.qqplot(resid3,fit=True,line='45')
#fitted v. residuals
fig, ax = plt.subplots()
_ = ax.scatter(fitted3, resid3)
# %%
# Breusch-Pagan Test for Heteroscedasticity
bptest = sms.diagnostic.het_breuschpagan(resid3,model3.model.exog)
# White Test for Heteroscedasticity, including squares and cross-product of exog
white_test = sms.diagnostic.het_white(resid3,model3.model.exog)
# %%
print('Breusch-Pagan Test and White LM Tests for Heteroscedasticity')
df3 = pd.DataFrame({'Test Type':['Breusch-Pagan Test', 'White Test'],
                   'Chi-Sq':[bptest[0], white_test[0]], 'DF':[2, 4],
                   'Prob>Chi-Sq':[bptest[1], white_test[1]]})
print(df3)

# %%
# create log tranforms
df["lncmpg"] = np.log(df["Cmpg"])
df["lneng"] = np.log(df["Eng"])
# %%
model4 = smf.ols(formula = 'lncmpg ~ lneng', data = df).fit()
print(model4.summary())
# %%
resid4 = model4.resid
fitted4 = model4.fittedvalues
# %%
#QQ plot of residuals
fig1 = sm.qqplot(resid4,fit=True,line='45')
#fitted v. residuals
fig, ax = plt.subplots()
_ = ax.scatter(fitted4, resid4)
# %%
