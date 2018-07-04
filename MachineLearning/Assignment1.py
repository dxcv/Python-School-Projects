
# coding: utf-8

import pandas as pd
import pandas_datareader as pdr
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


msft = pdr.DataReader('MSFT', 'iex', start = '2017-05-01', end = '2018-04-30')
msft = msft.rename(index = {i: pd.datetime(int(i[:4]), int(i[5:7]), int(i[8:10])) for i in msft.index.values.tolist()})
msft['seq'] = range(msft.shape[0])

#-----------------------------------
# Question 1
x1 = np.column_stack((msft.shift(1)['close'], msft.shift(2)['close'], msft.shift(3)['close'],
                                         msft.shift(4)['close'], msft.shift(5)['close']))[5:]
y1 = msft['close'][5:]
train = np.array([True] * 227 + [False] * 20)
x1_train, x1_test, y1_train, y1_test = x1[train], x1[~train], y1[train], y1[~train]

x2 = np.column_stack((x1, msft['seq'][5:]))
y2 = msft['close'][5:]
x2_train, x2_test, y2_train, y2_test = x2[train], x2[~train], y2[train], y2[~train]

ma = msft.apply(lambda x: x.shift(1).rolling(5).mean()).dropna()['close']
x3 = np.column_stack((x2, ma))
y3 = msft['close'][5:]
x3_train, x3_test, y3_train, y3_test = x3[train], x3[~train], y3[train], y3[~train]


# Build model here
output = pd.DataFrame()

regr1 = linear_model.LinearRegression()
regr1.fit(x1_train, y1_train)

regr2 = linear_model.LinearRegression()
regr2.fit(x2_train, y2_train)

regr3 = linear_model.LinearRegression()
regr3.fit(x3_train, y3_train)

output.loc['RMSE on training','Model 1'] = np.sqrt(metrics.mean_squared_error(y1_train, regr1.predict(x1_train)) )
output.loc['RMSE on training','Model 2'] = np.sqrt(metrics.mean_squared_error(y2_train, regr2.predict(x2_train)) )
output.loc['RMSE on training','Model 3'] = np.sqrt(metrics.mean_squared_error(y3_train, regr3.predict(x3_train)) )

output.loc['RMSE on test','Model 1'] = np.sqrt(metrics.mean_squared_error(y1_test, regr1.predict(x1_test)) )
output.loc['RMSE on test','Model 2'] = np.sqrt(metrics.mean_squared_error(y2_test, regr2.predict(x2_test)) )
output.loc['RMSE on test','Model 3'] = np.sqrt(metrics.mean_squared_error(y3_test, regr3.predict(x3_test)) )

# Show results
print(output)


#---------------------------------------------------
# Question 2
def my_discretize(m, step):
    n = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j] > 0.0:
                n[i,j] = np.ceil(m[i,j] / step)
            elif m[i,j] < 0.0:
                n[i,j] = -np.ceil(-m[i,j] / step)
    return n

diff = np.zeros((msft.shape[0] - 5, 5))
for i in range(5):
    diff[:,i] = msft.apply(lambda x: x.shift(i) - x.shift(i + 1))['close'][5:]

# Build model here
output2 = pd.DataFrame()
for step in (1, 0.5, 0.2):
    x = np.column_stack((x3, my_discretize(diff, step)))
    y = msft['close'][5:]
    x_train, x_test, y_train, y_test = x[train], x[~train], y[train], y[~train]
    regr_step = linear_model.LinearRegression()
    regr_step.fit(x_train, y_train)
    output2.loc['RMSE on training','Model with step = '+str(step)] = round(np.sqrt(metrics.mean_squared_error(y_train, regr_step.predict(x_train))),3) 
    output2.loc['RMSE on test','Model with step = '+str(step)] = round(np.sqrt(metrics.mean_squared_error(y_test, regr_step.predict(x_test))) ,3)

# Show results
print(output2)

#--------------------------------------------
# Question 3

output3 = pd.DataFrame()

i=0
fig = plt.figure(figsize=(18,11))  
for step in (1, 0.5, 0.2):
    
    i+=1
    
    x = np.column_stack((x3, my_discretize(diff, step)))
    y = msft['close'][5:]
    x_train, x_test, y_train, y_test = x[train], x[~train], y[train], y[~train]
    
    alpha_list = []
    rmse_training_list = []
    rmse_test_list = []
    complexity_list = []
    
    for a in (0.001, 0.01, 0.1, 1, 10):
        clf = linear_model.Ridge(alpha=a)
        clf.fit(x_train, y_train) 
        rmse_training = np.sqrt(metrics.mean_squared_error(y_train, clf.predict(x_train)))
        rmse_test = np.sqrt(metrics.mean_squared_error(y_test, clf.predict(x_test)))
        complexity = np.sqrt(sum(np.square(clf.coef_))+np.square(clf.intercept_))
          
        alpha_list.append(a)
        rmse_training_list.append(rmse_training)
        rmse_test_list.append(rmse_test)
        complexity_list.append(complexity)
        
        output3.loc['step='+str(step)+' RMSE on training','alpha='+str(a)] = rmse_training
        output3.loc['step='+str(step)+' RMSE on test','alpha='+str(a)] = rmse_test
        output3.loc['step='+str(step)+' model complexity','alpha='+str(a)] = complexity

    ax1 = fig.add_subplot(2,2,i)
    ax1.set_xscale('log')
    ax1.plot(alpha_list,rmse_training_list,label='RMSE on Training',c='r')
    ax1.plot(alpha_list,rmse_test_list,label='RMSE on Test',c='y')
    ax1.legend(['RMSE on Training','RMSE on Test'],loc='center left')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Alpha')
    ax1.grid(True)
    ax1.set_title('Step = '+str(step),fontsize = 13)
    
    ax2 = ax1.twinx()
    ax2.set_xscale('log')# this is the important function
    ax2.plot(alpha_list,complexity_list,c='b')
    ax2.legend(['Complexity'])
    ax2.set_ylabel('Model Complexity')
    ax2.set_xlabel('Alpha')

fig.show()

print(output3)
