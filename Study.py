import matplotlib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, KFold, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import *
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE
import xgboost as xgb
from scipy.stats import randint, expon, reciprocal, uniform
import time


def studyMissingValuesVIP(df: DataFrame):
    sns.countplot(data=df, x="VIP", hue="Transported")
    plt.title("Nombre de Passager transporté en fonction du VIP")
    plt.show()
    df["VIP"], uniques = pd.factorize(df["VIP"])
    print(abs(df["VIP"].corr(df["Transported"])))


def studyMissingValuesSide(df: DataFrame):
    groupSide = (df.groupby(["Group", "Side"])["Side"].size().unstack().fillna(0) > 0).sum(axis=1)
    surnameSide = (df[df["Group_size"] > 1].groupby(["Surname", "Side"])["Side"].size().unstack().fillna(0) > 0).sum(axis=1)
    sideSolo = df[df["Solo"] == 1]

    sns.barplot(x=groupSide.value_counts().index, y=groupSide.value_counts().values)
    plt.title("Nombre de Side par Groupe")
    plt.show()

    sns.barplot(x=surnameSide.value_counts().index, y=surnameSide.value_counts().values)
    plt.title("Nombre de Side par Surname")
    plt.show()

    sns.countplot(sideSolo, x="Side")
    plt.title("Répartition des Side pour les passagers voyageant seul")
    plt.show()


def studyMissingValuesDeck(df: DataFrame):
    df[np.array(["Deck", "Num", "Side"])] = df["Cabin"].str.split('/', expand=True)
    df[np.array(["Group", "NbInGroup"])] = df["PassengerId"].str.split('_', expand=True)
    df[np.array(["FirstName", "Surname"])] = df["Name"].str.split(" ", expand=True)

    df["Group_size"] = df["Group"].map(lambda x: df["Group"].value_counts()[x])
    df["Solo"] = (df["Group_size"] == 1).astype(int)


    deckDestination = df.groupby(["Destination", "Deck"])["Deck"].size().unstack().fillna(0)
    sns.heatmap(deckDestination, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    deckHomePlanet = df.groupby(["HomePlanet", "Deck"])["Deck"].size().unstack().fillna(0)
    sns.heatmap(deckHomePlanet, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    deckGroup = (df.groupby(["Group", "Deck"])["Deck"].size().unstack().fillna(0) > 0).sum(axis=1)
    sns.barplot(x=deckGroup.value_counts().index, y=deckGroup.value_counts().values)
    plt.title("Nombre de deck par groupe")
    plt.show()

    surnameGroup = (df.groupby(["Surname", "Deck"])["Deck"].size().unstack().fillna(0) > 0).sum(axis=1)
    sns.barplot(x=surnameGroup.value_counts().index, y=surnameGroup.value_counts().values)
    plt.title("Nombre de deck par Surname")
    plt.show()

    SoloGroup = df.groupby(["HomePlanet", "Solo", "Deck"])["Deck"].size().unstack().fillna(0)
    sns.heatmap(SoloGroup, annot=True, fmt='g', cmap='coolwarm')
    plt.show()


def studyMissingValuesDestination(df: DataFrame):
    df[np.array(["FirstName", "Surname"])] = df["Name"].str.split(' ', expand=True)
    destinationSurname = (df.groupby(["Surname", "Destination"])["Destination"].size().unstack().fillna(0) > 0).sum(axis=1)

    sns.barplot(x=destinationSurname.value_counts().index, y=destinationSurname.value_counts().values)
    plt.title("Nombre de destination par Surname")
    plt.show()

    df[np.array(["Deck", "Num", "Side"])] = df["Cabin"].str.split('/', expand=True)
    destinationDeck = df.groupby(["Deck", "Destination"])["Destination"].size().unstack().fillna(0)
    sns.heatmap(destinationDeck, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    df[np.array(["Group", "NbInGroup"])] = df["PassengerId"].str.split('_', expand=True)
    destinationGroup = (df.groupby(["Group", "Destination"])["Destination"].size().unstack().fillna(0) > 0).sum(axis=1)
    sns.barplot(x=destinationGroup.value_counts().index, y = destinationGroup.value_counts().values)
    plt.title("Nombre de destination par Groupe")
    plt.show()

    sns.countplot(df, x="Destination", hue="Transported")
    plt.show()


def studyMissingValuesHomePlanet(df: DataFrame):
    df[np.array(["Group", "NbInGroup"])] = df["PassengerId"].str.split('_', expand=True)

    homePlanetGroup = df.groupby(["Group", "HomePlanet"])
    nombreDeHomePlanetDifferentesParGroupe = (homePlanetGroup["HomePlanet"].size().unstack().fillna(0) > 0).sum(axis=1)
    groupesPlusDe1HomePlanet = nombreDeHomePlanetDifferentesParGroupe.loc[nombreDeHomePlanetDifferentesParGroupe > 1]
    print("Nombre de groupes avec plus de 1 HomePlanet : " + str(len(groupesPlusDe1HomePlanet)))
    sns.countplot(nombreDeHomePlanetDifferentesParGroupe)
    plt.title('Nombre de HomePlanet par Group')
    plt.show()

    df[np.array(["FirstName", "Surname"])] = df["Name"].str.split(' ', expand=True)

    homePlanetSurname = df.groupby(["Surname", "HomePlanet"])
    nombreDeHomePlanetDifferentesParSurname = (homePlanetSurname["Surname"].size().unstack().fillna(0) > 0).sum(axis=1)
    surnamePlusDe1HomePlanet = nombreDeHomePlanetDifferentesParSurname.loc[nombreDeHomePlanetDifferentesParSurname > 1]

    print("Nombre de Surname avec plus de 1 HomePlanet : " + str(len(surnamePlusDe1HomePlanet)))
    sns.countplot(nombreDeHomePlanetDifferentesParSurname)
    plt.title('Nombre de HomePlanet par Surname')
    plt.show()



def studyMissingValues(df: DataFrame):
    df = df.copy(deep=True)
    # studyMissingValuesHomePlanet(df)
    # studyMissingValuesDestination(df)
    # studyMissingValuesDeck(df)
    # studyMissingValuesSide(df)
    studyMissingValuesVIP(df)


def showBillWithCryo(train : DataFrame):
    fig = plt.figure(figsize=(10, 20))
    bill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    for i, name in enumerate(bill):
        ax = fig.add_subplot(5, 2, 2 * i + 1)
        print("Nombre de personne en CryoSleep qui ont dépensé pour " + name + ": " + str(
            len(train[(train["CryoSleep"] == True) & (train[name] > 0)])))
        sns.barplot(data=train, x="CryoSleep", ax=ax, y=name, errwidth=0)
        ax.set_title(name)
    fig.tight_layout()
    plt.show()

def showBillWithTransported(train : DataFrame):
    bill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    fig = plt.figure(figsize=(10, 15))
    for i, name in enumerate(bill):
        ax = fig.add_subplot(3,2, i+1)
        sns.histplot(data=train, x=name, bins=20, axes=ax, kde=True, hue="Transported")
        plt.xlim([0,4000])
        plt.ylim([0,2000])
        ax.set_title(name)
        plt.subplots_adjust(hspace=0.5)
    plt.show()

def showDeckTransported(train : DataFrame):
    trainC = train.copy()
    trainC[np.array(["Deck", "Num", "Side"])] = trainC["Cabin"].str.split('/', expand=True)
    trainC.drop("Cabin", axis=1, inplace=True)
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(5, 2, 1)
    sns.countplot(data=trainC, x="Side", ax=ax, hue="Transported")

    ax = fig.add_subplot(5, 2, 3)
    sns.countplot(data=trainC, x="Deck", ax=ax, hue="Transported", order=["A", "B", "C", "D", "E", "F", "G", "T"])
    fig.tight_layout()
    plt.show()

def showAgeWithTransported(train : DataFrame):
    plt.figure(figsize=(10, 4))
    sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
    plt.xlabel('Age')
    plt.show()
