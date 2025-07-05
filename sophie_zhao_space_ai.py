# ----------------------------- READ ME -------------------------------------#
# FIRST: Edit the path to the csv file where it says pd.read_csv after downloading the csv file. It currently does not have a path since the code has been done on Replit where the csv file was uploaded. 
#If a spacecraft encounters a solar flare, there is the possibility of severe damage to the spacecrafta. The included dataset is a dummy dataset. The parameters include frequency of extreme ultraviolet flashes (Hz), the brightness of a local area of the Sun (Cd) directly in front of the spacecraft, and the amount of change in solar magnetic flux per second (Wb/s). These factors contribute to the likelihood of a solar flare. Note that the values in the dummy dataset are arbitrary and are not necessarily close to the actual values of the parameters. If there is a risk of a solar flare, the spacecraft will temporarily change its trajectory to avoid the Sun.

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

#Create a variable to read the dataset
df = pd.read_csv("solar_flare_risk.csv")  #Edit the path to the csv file after downloading the csv file


#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
X = df.drop("Solar flare/CME in next 2-6 hours?", axis=1)
y = df["Solar flare/CME in next 2-6 hours?"]
X_train, X_test, y_train, y_test = train_test_split(X, y)


#Label encode the dataset
from sklearn import preprocessing

def labelEncoder(df, colsList): #Documentation from Girls Who Code
    le = preprocessing.LabelEncoder()
    for col in colsList:
        df[col] = le.fit_transform(df[col])
    return df
    
df = labelEncoder(df, ["Solar flare/CME in next 2-6 hours?","Frequency of extreme ultraviolet flashes (Hz)","Brighness of local area on Sun (Cd)","Change in solar magnetic flux (Wb/s)"])

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced")
clf.fit(X_train, y_train)



#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predictions)

print("The accuracy with the training data set of the Decision Tree is: " + str(train_acc))


#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions)

print("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))

#Print decision tree
from sklearn import tree #Documentation from Girls Who Code
def printTree(clf, cols):
    text_representation = tree.export_text(clf, feature_names = cols, class_names = ['No solar flare risk', 'Risk of solar flare'], show_weights = True)
    print(text_representation)

print("\nHere is the Decision Tree:\n")
printTree(clf, X.columns)