##########
# Part 1 #
##########

import numpy as np
from sklearn import datasets, metrics, feature_extraction, naive_bayes

data_train = datasets.fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes'))
data_test = datasets.fetch_20newsgroups(subset = 'test', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes'))
categories = data_train.target_names
target_map = {}
for i in range(len(categories)):
    if 'comp.' in categories[i]:
        target_map[i] = 0
    elif 'rec.' in categories[i]:
        target_map[i] = 1
    elif 'sci.' in categories[i]:
        target_map[i] = 2
    elif 'misc.forsale' in categories[i]:
        target_map[i] = 3
    elif 'talk.politics' in categories[i]:
        target_map[i] = 4
    else:
        target_map[i] = 5

tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x_train = tfidf_vectorizer.fit_transform(data_train.data)
x_test = tfidf_vectorizer.transform(data_test.data)
y_train = [target_map[i] for i in data_train.target]
y_test = [target_map[i] for i in data_test.target]


##########
# Part 1 #
##########

from sklearn import linear_model
from sklearn import model_selection, metrics
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd


def my_softmax (x, coef, intercept):
    class_number = coef.shape[0]
    data_point_number = x.shape[0]
    probability_all_list = []

    for data_point_n in range(data_point_number):
        single_x_array = x[data_point_n]
        sum_list = []
        probablity_list = []
        for class_n in range(class_number):
            sum_for_single_class = coef[class_n]*single_x_array.T+intercept[class_n]
            sum_for_single_class = np.exp(sum_for_single_class)
            sum_list.append(sum_for_single_class)
        sum_temp = np.array(sum_list).sum()
        for class_n in range(class_number):
            probablity_list.append(sum_list[class_n]/sum_temp)      
        probability_all_list.extend(probablity_list)
    probability_matrix = np.array(probability_all_list).reshape(data_point_number,class_number)

    return probability_matrix


output = pd.DataFrame()
c_temp_list = [0.01,0.1,1,10,100]
for c_temp in c_temp_list:
    logit = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', C = c_temp )
    logit.fit(x_train,y_train)
    y_test_predict = logit.predict(x_test)
    precision, recall, f1_score, support = score(y_test, y_test_predict, average = 'weighted')
    
    output.loc['Precision','C='+str(c_temp)] = precision
    output.loc['Recall','C='+str(c_temp)] = recall
    output.loc['F1 score','C='+str(c_temp)] = f1_score
print(output)


output2 = pd.DataFrame()
for c_temp in c_temp_list:
    logit = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', C = c_temp )
    logit.fit(x_train,y_train)
    my_predict_proba = my_softmax(x_test, logit.coef_, logit.intercept_)
    diff = np.linalg.norm(my_predict_proba - logit.predict_proba(x_test))
    output2.loc['Difference from Probabilities','C='+str(c_temp)] = diff
print(output2)


##########
# Part 2 #
##########

def top_words(logit,words_list,k):
    output = pd.DataFrame()
    class_number  = logit.coef_.shape[0] 
    for class_n in range(class_number):
        class_list = list(logit.coef_[class_n])
        number_position_list = []
        for number in class_list:
            position = class_list.index(number)
            number_position_list.append((number,position))
        number_position_list.sort(reverse=True)
        number_position_list = number_position_list[:k]
        for j in range(k):
            p = number_position_list[j][1]
            output.loc[str(j+1),'Category '+str(class_n+1)] = words_list[p]
    return output

logit = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', C = c_temp )
logit.fit(x_train,y_train)
words_list = tfidf_vectorizer.get_feature_names()
k=10
output3 = top_words(logit,words_list,k)
print(output3)


##########
# Part 3 #
##########
from sklearn import decomposition

tSVD = decomposition.TruncatedSVD(n_components = 100, n_iter = 20, random_state = 2018)
xr_train = tSVD.fit_transform(x_train)
xr_test = tSVD.transform(x_test)

##########
# Part 4 #
##########
import time
from sklearn import svm
parameters = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['poly'], 'degree': [2, 3], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'gamma': ['auto', 0.5, 1, 2], 'C': [0.1, 1, 10, 100]}]


start_1 = time.time()
clf_1 = model_selection.GridSearchCV(svm.SVC(), parameters)
clf_1.fit(x_train, y_train)
predicted_1 = clf_1.predict(x_test)
precision_1, recall_1, f1_score_1, support_1 = score(y_test, predicted_1, average = 'weighted')
end_1 = time.time()
print('Original Data')
print('Precision: '+str(precision_1))
print('Recall: '+str(recall_1))
print('f1_score: '+str(f1_score_1))
print('best score:'+str(clf_1.best_score_))
print('best parameters: '+str(clf_1.best_params_))
print('Seconds needed: '+str(end_1-start_1))


start_2 = time.time()
clf_2 = model_selection.GridSearchCV(svm.SVC(), parameters)
clf_2.fit(xr_train, y_train)
predicted_2 = clf_2.predict(xr_test)
precision_2, recall_2, f1_score_2, support_2 = score(y_test, predicted_2, average = 'weighted')
end_2 = time.time()

print('Reduced Data')
print('Precision: '+str(precision_2))
print('Recall: '+str(recall_2))
print('f1_score: '+str(f1_score_2))
print('best score:'+str(clf_2.best_score_))
print('best parameters: '+str(clf_2.best_params_))
print('Seconds needed: '+str(end_2-start_2))


##########
# Part 5 #
##########
from sklearn import tree, svm, ensemble

def bagging(x_train,y_train,x_test,y_test):
    bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators=50,  max_samples = 0.5, max_features = 1, oob_score = True, random_state = 2018)
    bagging.fit(x_train, y_train)
    predicted = bagging.predict(x_test)

    precision, recall, f1_score, support = score(y_test, predicted, average = 'weighted')
    return f1_score

def random_forest(x_train,y_train,x_test,y_test):
    model = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 20, max_features = 1, oob_score = True, random_state = 2018)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    precision, recall, f1_score, support = score(y_test, predicted, average = 'weighted')
    return f1_score

def adaboost(x_train,y_train,x_test,y_test):
    model = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    precision, recall, f1_score, support = score(y_test, predicted, average = 'weighted')
    return f1_score

def gboost(x_train,y_train,x_test,y_test):
    model = ensemble.GradientBoostingClassifier(n_estimators = 50, max_depth = 20, random_state = 2018)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    precision, recall, f1_score, support = score(y_test, predicted, average = 'weighted')
    return f1_score

output5 = pd.DataFrame()
x_train_data = [x_train,xr_train]
x_test_data = [x_test,xr_test]
cols = ['Using Original Data','Using Reduced Data']

for i in range(2):
    col = cols[i]
    x_1 = x_train_data[i]
    x_2 = x_test_data[i]
    output5.loc[col,'Bagging'] = bagging(x_1,y_train,x_2,y_test)
    output5.loc[col,'Random Forest'] = random_forest(x_1,y_train,x_2,y_test)
    output5.loc[col,'AdaBoost'] = adaboost(x_1,y_train,x_2,y_test)
    output5.loc[col,'Gradient Boosting'] = gboost(x_1,y_train,x_2,y_test)
print(output5)