from missingValues import missingValues
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler



# Calcul de correlations avec l'age
def findAgeIntervals(train):
    train.loc[train["Age"].isna(), "Age"] = train["Age"].median()
    prevAge = 0
    currentAge = 2
    currentInterval = train.loc[(train["Age"] >= prevAge) & (train["Age"] < currentAge)]
    prevCorr = abs(currentInterval["Age"].corr(currentInterval["Transported"]))
    liste = [0]
    currentEmptySize = 0

    while (len(currentInterval) > 0 or currentEmptySize < 10):
        currentAge += 1
        currentInterval = train.loc[(train["Age"] >= prevAge) & (train["Age"] < currentAge)]
        if (len(currentInterval) == 0):
            currentEmptySize += 1
            continue
        else:
            currentEmptySize = 0
        cor = abs(currentInterval["Age"].corr(currentInterval["Transported"]))

        if (prevCorr - cor >= 0.025 or prevCorr == cor):
            print("Change interval   :  pa " + str(prevAge) + " ca " + str(currentAge) + "     " + str(
                prevCorr - cor) + "   " + str(prevCorr) + "   " + str(cor))
            liste.append(currentAge - 1)
            prevAge = currentAge - 1
            currentAge += 1
            currentInterval = train.loc[(train["Age"] >= prevAge) & (train["Age"] < currentAge)]
            cor = abs(currentInterval["Age"].corr(currentInterval["Transported"]))

        prevCorr = cor

    print(liste)
    s = train.loc[(train["Age"] >= 0) & (train["Age"] < 3)]
    s2 = train.loc[(train["Age"] >= 0) & (train["Age"] < 4)]
    print(abs(s["Age"].corr(s["Transported"])))
    print(abs(s2["Age"].corr(s2["Transported"])))
    print(abs(s["Age"].corr(s["Transported"])) - abs(s2["Age"].corr(s2["Transported"])))

    return liste

def dropRemainingMissingValues(train):
    print("Remaining missing values :  " + str(len(train[train.isna().sum(axis=1)>0])) + " / " + str(len(train)))
    train.drop(train[train.isna().sum(axis=1) > 0].index, inplace=True)
    print("After :  " + str(len(train)))


def createAgeGroup(df: DataFrame, train):
    print("Age missing values: ", df["Age"].isna().sum())
    df.loc[df["Age"].isna(), "Age"] = df["Age"].median()
    print("Age missing values: ", df["Age"].isna().sum())
    liste = findAgeIntervals(train)
    df["Age_group"] = np.nan
    prevAge = 0
    for i in range(1, len(liste)):
        df.loc[(df['Age'] >= prevAge) & (df['Age'] <= liste[i]), "Age_group"] = i - 1
        prevAge = liste[i]

def createLuxeBasic(df: DataFrame):
    #showBillWithTransported()
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["Basics"] = df["ShoppingMall"] + df["FoodCourt"]

    """corr_l = df["Luxury"].corr(df["Transported"])
    corr_b = df["Basics"].corr(df["Transported"])
    print("Correlation entre Luxury et transported : " + str(corr_l))
    print("Correlation entre Basics et transported : " + str(corr_b))"""

def createNoBill(df: DataFrame):
    df["NoBill"] = (df[np.array(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"])].sum(axis=1) == 0).astype(int)
    """print(df.head())
    sns.countplot(data=df, x="NoBill", hue="Transported")
    plt.title("Nombre de personne transporté en fonction de si ils ont dépensé")
    plt.show()
    print(df["NoBill"].corr(df["Transported"]))"""


def createSolo(df: DataFrame):
    df["Group_size"] = df["Group"].map(lambda x: df["Group"].value_counts()[x])
    """print(df["Group_size"].corr(df['Transported']))
    sns.countplot(data=df, x="Group_size", hue="Transported")
    plt.title("Nombre de personne transporté en fonction de la taille de leur groupe")
    # plt.show()"""

    df["Solo"] = (df["Group_size"] == 1).astype(int)
    """print(df["Solo"].corr(df['Transported']))
    sns.countplot(data=df, x="Solo", hue="Transported")
    plt.title("Nombre personne voyagant seul transporté")
    plt.show()"""


def createDummies(df: DataFrame):
    # planetes
    homePlanete = pd.get_dummies(df["HomePlanet"])
    df.drop("HomePlanet", axis=1, inplace=True)

    # Side
    sides = pd.get_dummies(df["Side"])
    df.drop("Side", axis=1, inplace=True)

    # Destination
    destination = pd.get_dummies(df["Destination"])
    df.drop("Destination", axis=1, inplace=True)

    # Deck
    deck = pd.get_dummies(df["Deck"])
    df.drop("Deck", axis=1, inplace=True)

    return [homePlanete, sides, destination, deck]

def separateColumns(df: DataFrame):
    # Split la colonne cabin en les colonnes Deck, Num et Side
    df[np.array(["Deck", "Num", "Side"])] = df["Cabin"].str.split('/', expand=True)
    df.drop("Cabin", axis=1, inplace=True)
    df.drop("Num", axis=1, inplace=True)

    # Split la colonne PassengerId en les colonnes Group et NbInGroup
    df[np.array(["Group", "NbInGroup"])] = df["PassengerId"].str.split('_', expand=True)
    df.drop("PassengerId", axis=1, inplace=True)
    df.drop("NbInGroup", axis=1, inplace=True)

    df[np.array(["FirstName", "Surname"])] = df["Name"].str.split(' ', expand=True)
    df.drop("Name", axis=1, inplace=True)

def dropVip(df):
    df.drop("VIP", axis=1, inplace=True)

def handleCategorical(df: DataFrame):
    df["VIP"], uniques = pd.factorize(df["VIP"])
    df["CryoSleep"], uniques = pd.factorize(df["CryoSleep"])

def dropColumns(df: DataFrame):
    df.drop("Group", axis=1, inplace=True)
    df.drop("FirstName", axis=1, inplace=True)
    df.drop("Surname", axis=1, inplace=True)
    df.drop("Group_size", axis=1, inplace=True)
    df.drop("Age", axis=1, inplace=True)

def preprocessing(df, train, test = False):
    separateColumns(df)
    createNoBill(df)
    createSolo(df)
    createLuxeBasic(df)
    createAgeGroup(df, train)
    missingValues(df)
    if(not test):
        dropRemainingMissingValues(df)
    handleCategorical(df)
    homePlanete, sides, destination, deck = createDummies(df)
    dropColumns(df)
    preproDf = pd.concat([df, homePlanete, destination, sides, deck], axis=1)
    for column in preproDf:
        preproDf[column] = MinMaxScaler().fit_transform(np.array(preproDf[column]).reshape(-1, 1))
    return preproDf
