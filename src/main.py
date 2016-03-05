# -*-coding: utf-8 -*-
"""
author: young
create on: 2016/2/26
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import svm

# Cleaning the dataset
def harmonize_data(titanic):
    # Filling the blank data
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    # Assigning binary form to data foe calculation
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 0
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    return titanic


# Creating submission file
def create_submission(rfc, train, test, predictors, filename):

    rfc.fit(train[predictors], train["Survived"])
    predictions = rfc.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    #draw_feature_importance(predictors, rfc.feature_importances_)

    submission.to_csv(filename, index=False)

# Draw feature's importance
def draw_feature_importance(X,importances):

    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(X)), importances[indices], color="r", align="center")

    plt.show()


def random_forest(train_data, predictors, test_data):
    #Applying method
    max_score = 0
    best_n = 0
    for n in range(1,100):
        rfc_scr = 0.
        rfc = RandomForestClassifier(n_estimators=n)
        for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
            rfc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
            rfc_scr += rfc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
        if rfc_scr > max_score:
            max_score = rfc_scr
            best_n = n

    print(best_n, max_score)
    rfc = RandomForestClassifier(best_n)
    # Creating submission
    create_submission(rfc, train_data, test_data, predictors, "rfcsurvivors.csv")


def extreamly_random_forest(train_data, predictors):
    # Applying method
    max_score = 0
    best_n = 0
    for n in range(1, 100):
        rfc_scr = 0.
        rfc = ExtraTreesClassifier(n_estimators=n)
        for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
            rfc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
            rfc_scr += rfc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
        if rfc_scr > max_score:
            max_score = rfc_scr
            best_n = n

    print(best_n, max_score)
    rfc = ExtraTreesClassifier(best_n)

    # Creating submission
    create_submission(rfc, train_data, test_data, predictors, "rfcsurvivors.csv")


def svm_clf(train, test, predictors_attri):
    clf = svm.SVC(C=1.0)
    create_submission(clf, train, test, predictors_attri, "rfcsurvivors.csv")


def women_survided(titanic):
    pass


if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv', dtype={"Age": np.float64},)
    test_df = pd.read_csv('../data/test.csv', dtype={"Age": np.float64},)
    # Defining the clean dataset
    train_data = harmonize_data(train_df)
    test_data = harmonize_data(test_df)

    # Feature engineering
    #train_data["SP"] = (train_data["SibSp"]*train_data["Parch"])**2
    #train_data["PA"] = train_data["Pclass"]*train_data["Age"]

    test_data["SP"] = (test_data["SibSp"]*test_data["Parch"])**2
    test_data["PA"] = test_data["Pclass"]*test_data["Age"]

    # Defining predictor
    predictors = ["Pclass", "Sex", "Age"]
    #extreamly_random_forest(train_data)
    svm_clf(train_data, test_data,predictors)


