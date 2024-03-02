from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split

import pandas as pd
import openpyxl
from sklearn.tree import DecisionTreeClassifier

def printScore(predictions):
    print('Accuracy:', accuracy_score(y_test, predictions))
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    print('Roc:', roc_auc)
    print('F1:', f1_score(y_test, predictions))

path = 'Datasets/'

data = pd.read_excel(path + 'Credit_card.xlsx', header = 0)
training_data, test_data = train_test_split (data, test_size=0.3, random_state=50, shuffle = True)
X_train = training_data.iloc[:,1:-2]
y_train = training_data.iloc[:,-1]
X_test = test_data.iloc[:,1:-2]
y_test = test_data.iloc[:,-1]

print('Dane nie zbalansowane')
print('DT:')
tree_clf = DecisionTreeClassifier(max_depth=5, criterion='gini')
tree_clf.fit(X_train, y_train)
predictions = tree_clf.predict(X_test)
printScore(predictions)

print('----------------------------------------------------------------')
print('RF:')
rf = RandomForestClassifier(n_estimators = 10, criterion = 'log_loss', max_depth = 5)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
printScore(predictions)

print('===============================================================')
print('Dane zbalansowane przez under-sumpling')

rus = RandomUnderSampler(random_state=420)
X_balanced_training, y_balanced_training = rus.fit_resample(X_train, y_train)
print('DT:')
tree_clf = DecisionTreeClassifier(max_depth=4, criterion='gini')
tree_clf.fit(X_balanced_training, y_balanced_training)
predictions = tree_clf.predict(X_test)
printScore(predictions)

print('----------------------------------------------------------------')
print('RF:')
rf = RandomForestClassifier(n_estimators = 10, criterion = 'log_loss', max_depth = 4)
rf.fit(X_balanced_training, y_balanced_training)
predictions = rf.predict(X_test)
printScore(predictions)

print('===============================================================')
print('Dane zbalansowane przez over-sumpling')

ros = RandomOverSampler(random_state=100)
X_balanced_training, y_balanced_training = ros.fit_resample(X_train, y_train)
print('DT:')
tree_clf = DecisionTreeClassifier(max_depth=4, criterion='gini')
tree_clf.fit(X_balanced_training, y_balanced_training)
predictions = tree_clf.predict(X_test)
printScore(predictions)

print('----------------------------------------------------------------')
print('RF:')
rf = RandomForestClassifier(n_estimators = 10, criterion = 'log_loss', max_depth = 4)
rf.fit(X_balanced_training, y_balanced_training)
predictions = rf.predict(X_test)
printScore(predictions)

print('===============================================================')
print('Dane zbalansowane przez SMOTE')

smote = SMOTE(random_state=420)
X_balanced_training, y_balanced_training = smote.fit_resample(X_train, y_train)
print('DT:')
tree_clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
tree_clf.fit(X_balanced_training, y_balanced_training)
predictions = tree_clf.predict(X_test)
printScore(predictions)

print('----------------------------------------------------------------')
print('RF:')
rf = RandomForestClassifier(n_estimators = 10, criterion = 'log_loss', max_depth = 2)
rf.fit(X_balanced_training, y_balanced_training)
predictions = rf.predict(X_test)
printScore(predictions)