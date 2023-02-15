import matplotlib
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import *
import numpy as np

matplotlib.use('tkagg')

# Lecture des fichiers CSV
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')


# etudier les correlations
# imputer les données en fonction des groupes
# TODO faire des categories d'ages
# TODO faire dummy pour deck
# TODO drop les ligne avec 2 valeurs nan ou plus
# TODO meme groupe viennent meme planete
# TODO regarder si passager seul ou en famille plus sauvé si oui ajouter une colonne seul/famille
# TODO regarder les decks
# TODO matrice de confusion,
# TODO regarder importance des variables

# Calcul de correlations avec l'age
def findAgeIntervals(train):
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

    train["Age_group"] = np.nan
    prevAge = 0
    for i in range(1, len(liste)):
        train.loc[(train['Age'] >= prevAge) & (train['Age'] <= liste[i]), "Age_group"] = i - 1
        prevAge = liste[i]

    print(train["Age_group"].corr(train["Transported"]))


# Graphiques
def graph(data, x, y, type="strip"):
    if type == "strip":
        sns.stripplot(data=data, x=x, y=y, linewidth=0.1, s=1)
    if type == "count":
        data["t"] = data[x] + 2 * data[y]
        sns.countplot(data=data, x="t")
        data.drop("t", axis=1, inplace=True)

    plt.show()

def showBillWithCryo():
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

def showBillWithTransported():
    bill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    fig = plt.figure(figsize=(10, 15))
    for i, name in enumerate(bill):
        ax = fig.add_subplot(3,2, i+1)
        sns.histplot(data=train, x=name, bins=20, axes=ax, kde=True, hue="Transported")
        plt.xlim([0,4000])
        plt.ylim([0,2000])
        ax.set_title(name)
        plt.subplots_adjust(hspace=0.5)
    #fig.tight_layout()
    plt.show()

def showDeckTransported():
    trainC = train.copy()
    trainC[["Deck", "Num", "Side"]] = trainC["Cabin"].str.split('/', expand=True)
    trainC.drop("Cabin", axis=1, inplace=True)
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(5, 2, 1)
    sns.countplot(data=trainC, x="Side", ax=ax, hue="Transported")

    ax = fig.add_subplot(5, 2, 3)
    sns.countplot(data=trainC, x="Deck", ax=ax, hue="Transported", order=["A", "B", "C", "D", "E", "F", "G", "T"])
    fig.tight_layout()
    plt.show()

def showAgeWithTransported():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
    plt.xlabel('Age')
    plt.show()


#TODO Missing Values Preprocessing

def missingValuesBill(df: DataFrame):
    bill = np.array(["RoomService", "Spa", "ShoppingMall", "VRDeck", "FoodCourt"])
    print("Nombre valeurs manquantes bill :",df[bill].isna().sum().sum())
    for b in bill:
        df.loc[(df[b].isna()) & (df['CryoSleep'] == True), b] = 0
    print("Nombre valeurs manquantes bill :",df[bill].isna().sum().sum())

    for b in bill:
        df.loc[(df[b].isna()), b] = df[b].median()
    print("Nombre valeurs manquantes bill :", df[bill].isna().sum().sum())

    # Update NoBill Column
    df.loc[(df["RoomService"]+df["Spa"]+df["ShoppingMall"]+df["VRDeck"]+df["FoodCourt"] == 0), "NoBill"] = 1
    # Update Luxe/Basics
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["Basics"] = df["ShoppingMall"] + df["FoodCourt"]

def missingValuesHomePlanet(df: DataFrame):
    print(len(df[df['HomePlanet'].isna()]))
    linesMissingHomePlanet = df[df['HomePlanet'].isna()]

    print(df.iloc[5371])


    for index, row in linesMissingHomePlanet.iterrows():
        group = df.loc[df["Group"] == row["Group"], ["HomePlanet", "Group"]]
        homeplanet = group[~group["HomePlanet"].isna()].index
        if(len(homeplanet.values) > 0):
            df.at[index, "HomePlanet"] = group["HomePlanet"][homeplanet.values[0]]

    print(len(df[df['HomePlanet'].isna()]))

    linesMissingHomePlanet = df[df['HomePlanet'].isna()]

    for index, row in linesMissingHomePlanet.iterrows():
        surname = df.loc[df["Surname"] == row["Surname"], ["HomePlanet", "Surname"]]
        homeplanet = surname[~surname["HomePlanet"].isna()].index
        if(len(homeplanet.values) > 0):
            if(index == 5371):
                print(homeplanet.values)
            df.at[index, "HomePlanet"] = surname["HomePlanet"][homeplanet.values[0]]


    print(len(df[df['HomePlanet'].isna()]))
    print(df[df['HomePlanet'].isna()]["Deck"])


def missingValuesCryoSleep(df: DataFrame):
    print("CryoSleep missing values : " + str(df["CryoSleep"].isna().sum()))
    df.loc[(df["CryoSleep"].isna()) & (df["NoBill"] == 1), "CryoSleep"] = True
    df.loc[(df["CryoSleep"].isna()) & (df["NoBill"] == 0), "CryoSleep"] = False
    print("CryoSleep missing values : " + str(df["CryoSleep"].isna().sum()))

def missingValueVIP(df: DataFrame):
    print("VIP missing values : " + str(df["VIP"].isna().sum()))
    df.loc[(df["VIP"].isna()), "VIP"] = False
    print("VIP missing values : " + str(df["VIP"].isna().sum()))

def missingValueSide(df: DataFrame):
    missingSide = df.loc[(df["Side"].isna())]
    print("Side missing values: ", df["Side"].isna().sum())
    for index, row in missingSide.iterrows():
        group = df.loc[(df["Group"]==row["Group"]), ["Side","Group"]]
        side = group[~group["Side"].isna()].index
        if len(side.values) > 0:
            df.at[index, "Side"] = group["Side"][side.values[0]]
    print("Side missing values: ", df["Side"].isna().sum())

    missingSide = df.loc[(df["Side"].isna())]
    for index, row in missingSide.iterrows():
        group = df.loc[(df["Surname"] == row["Surname"]) & (df["Group_size"] > 1), np.array(["Side", "Surname"])]
        side = group[~group["Side"].isna()].index
        if len(side.values) > 0:
            # Met la valeur de Side la plus présente parmis le Surname
            df.at[index, "Side"] = group["Side"].value_counts().index[0]

    print("Side missing values: ", df["Side"].isna().sum())
    df.loc[(df["Side"].isna()), "Side"] = "P"
    print("Side missing values: ", df["Side"].isna().sum())


def createLuxeBasic(df: DataFrame):
    #showBillWithTransported()
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["Basics"] = df["ShoppingMall"] + df["FoodCourt"]

    """corr_l = df["Luxury"].corr(df["Transported"])
    corr_b = df["Basics"].corr(df["Transported"])
    print("Correlation entre Luxury et transported : " + str(corr_l))
    print("Correlation entre Basics et transported : " + str(corr_b))"""

def createNoBill(df: DataFrame):
    df["NoBill"] = (df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1) == 0).astype(int)
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

    sns.barplot(x=groupSide.value_counts().index, y = groupSide.value_counts().values)
    plt.title("Nombre de Side par Groupe")
    plt.show()

   
    sns.barplot(x=surnameSide.value_counts().index, y = surnameSide.value_counts().values)
    plt.title("Nombre de Side par Surname")
    plt.show()

    sns.countplot(sideSolo, x="Side")
    plt.title("Répartition des Side pour les passagers voyageant seul")
    plt.show()


def studyMissingValuesDeck(df: DataFrame):
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)
    df[["Group", "NbInGroup"]] = df["PassengerId"].str.split('_', expand=True)

    deckDestination = df.groupby(["Destination", "Deck"])["Deck"].size().unstack().fillna(0)
    sns.heatmap(deckDestination, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    deckHomePlanet = df.groupby(["HomePlanet", "Deck"])["Deck"].size().unstack().fillna(0)
    sns.heatmap(deckHomePlanet, annot=True, fmt='g', cmap='coolwarm')
    plt.show()


def studyMissingValuesDestination(df: DataFrame):
    df[["FirstName", "Surname"]] = df["Name"].str.split(' ', expand=True)
    destinationGroup = df.groupby(["Surname", "Destination"])["Destination"].size().unstack().fillna(0)
    sns.heatmap(destinationGroup, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)
    destinationDeck = df.groupby(["Deck", "Destination"])["Destination"].size().unstack().fillna(0)
    sns.heatmap(destinationDeck, annot=True, fmt='g', cmap='coolwarm')
    plt.show()

    sns.countplot(df, x="Destination", hue="Transported")
    plt.show()


def studyMissingValuesHomePlanet(df: DataFrame):
    df[["Group", "NbInGroup"]] = df["PassengerId"].str.split('_', expand=True)

    homePlanetGroup = df.groupby(["Group", "HomePlanet"])
    nombreDeHomePlanetDifferentesParGroupe = (homePlanetGroup["HomePlanet"].size().unstack().fillna(0) > 0).sum(axis=1)
    groupesPlusDe1HomePlanet = nombreDeHomePlanetDifferentesParGroupe.loc[nombreDeHomePlanetDifferentesParGroupe > 1]
    print("Nombre de groupes avec plus de 1 HomePlanet : " + str(len(groupesPlusDe1HomePlanet)))
    sns.countplot(nombreDeHomePlanetDifferentesParGroupe)
    plt.title('Nombre de HomePlanet par Group')
    plt.show()

    df[["FirstName", "Surname"]] = df["Name"].str.split(' ', expand=True)

    homePlanetSurname = df.groupby(["Surname", "HomePlanet"])
    nombreDeHomePlanetDifferentesParSurname = (homePlanetSurname["Surname"].size().unstack().fillna(0) > 0).sum(axis=1)
    surnamePlusDe1HomePlanet = nombreDeHomePlanetDifferentesParSurname.loc[nombreDeHomePlanetDifferentesParSurname > 1]

    print("Nombre de Surname avec plus de 1 HomePlanet : " + str(len(surnamePlusDe1HomePlanet)))
    sns.countplot(nombreDeHomePlanetDifferentesParSurname)
    plt.title('Nombre de HomePlanet par Surname')
    plt.show()


def missingValues(df):
    missingValuesHomePlanet(df)
    missingValuesBill(df)
    missingValuesCryoSleep(df)

def studyMissingValues(df: DataFrame):
    df = df.copy(deep=True)
    # studyMissingValuesHomePlanet(df)
    # studyMissingValuesDestination(df)
    # studyMissingValuesDeck(df)
    # studyMissingValuesSide(df)
    studyMissingValuesVIP(df)

def createDummies(df: DataFrame):
    # planetes
    homePlanete = pd.get_dummies(df["HomePlanet"])
    df.drop("HomePlanet", axis=1, inplace=True)

    # Side
    sides = pd.get_dummies(df["Side"])
    df.drop("Side", axis=1, inplace=True)

    destination = pd.get_dummies(df["Destination"])
    df.drop("Destination", axis=1, inplace=True)

    return [homePlanete, sides, destination]

def separateColumns(df: DataFrame):
    # Split la colonne cabin en les colonnes Deck, Num et Side
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)
    df.drop("Cabin", axis=1, inplace=True)

    # Split la colonne PassengerId en les colonnes Group et NbInGroup
    df[["Group", "NbInGroup"]] = df["PassengerId"].str.split('_', expand=True)
    df.drop("PassengerId", axis=1, inplace=True)
    df.drop("NbInGroup", axis=1, inplace=True)

    df[["FirstName", "Surname"]] = df["Name"].str.split(' ', expand=True)
    df.drop("Name", axis=1, inplace=True)


def preprocessing(df):
    # Initialise les colones numeriques et nominales
    """numerical = df.select_dtypes("float64", None)
    nominal = df.select_dtypes("object", None)

    # On remplace les données manquantes numériques
    numerical["Age"] = SimpleImputer(strategy="median").fit_transform(numerical[["Age"]])
    numerical["RoomService"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(
        numerical[["RoomService"]])
    numerical["FoodCourt"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["FoodCourt"]])
    numerical["ShoppingMall"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(
        numerical[["ShoppingMall"]])
    numerical["Spa"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["Spa"]])
    numerical["VRDeck"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["VRDeck"]])

    # Creation des tranches d'ages en fonction du graphe obtenu
    findAgeIntervals(train)

    numerical.drop("Age", axis=1, inplace=True)

    numerical["b_needs"] = numerical["FoodCourt"] + numerical["ShoppingMall"]
    numerical["l_needs"] = numerical["RoomService"] + numerical["Spa"] + numerical["VRDeck"]

    numerical.drop("FoodCourt", axis=1, inplace=True)
    numerical.drop("ShoppingMall", axis=1, inplace=True)
    numerical.drop("VRDeck", axis=1, inplace=True)
    numerical.drop("RoomService", axis=1, inplace=True)
    numerical.drop("Spa", axis=1, inplace=True)

    # nb de valeurs
    # print(train.nunique())

    # Remplace les strings par des int
    for column in nominal:
        nominal[column], uniques = pd.factorize(nominal[column])

    # Remplace les données manquantes
    for column in nominal:
        nominal[column] = SimpleImputer(missing_values=-1, strategy="most_frequent").fit_transform(nominal[[column]])

    newDf = pd.concat([numerical, nominal, homePlanete, destination, sides], axis=1)
    # Normalise les données
    for column in newDf:
        newDf[column] = MinMaxScaler().fit_transform(np.array(newDf[column]).reshape(-1, 1))
    return newDf
"""

# Random forest feature importante
def randomForest():
    print("feur")

def xGBoost():
    print("xgboost")

def SVM(train_process, test_process, y):
    # Spécification des paramètres de la grille de recherche
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01]
    }

    # Recherche des meilleurs paramètres
    search = GridSearchCV(SVC(), parameters, refit=True, verbose=3).fit(train_process, y)

    # Entraînement du modèle avec les meilleurs paramètres
    svc_best = SVC(**search.best_params_)
    svc_best.fit(train_process, y)

    # Prédiction des étiquettes pour les données de test
    pred_train = svc_best.predict(test_process)
    submit = pd.DataFrame({'PassengerId': test["PassengerId"], 'Transported': pred_train})
    submit.to_csv("./Data/submit.csv", index=False)


def Logistic(train_process, test_process, y):
    parameters = {'penalty': ['l1', 'l2'],
                  'C': [0.1, 1, 10, 100],
                  'solver': ['saga', 'liblinear'],
                  'max_iter': [100, 1000, 10000, 100000]
                  }
    Search = GridSearchCV(LogisticRegression(), parameters, scoring='accuracy', n_jobs=-1, cv=5, verbose=1).fit(
        train_process, y)

    log_model = LogisticRegression(penalty=Search.best_params_["penalty"], C=Search.best_params_["C"],
                                   max_iter=Search.best_params_["max_iter"], solver=Search.best_params_["solver"])
    log_model.fit(train_process, y)
    pred_train = log_model.predict(test_process)

    submit = pd.DataFrame({'PassengerId': test["PassengerId"], 'Transported': pred_train})
    submit.to_csv("./Data/submit.csv", index=False)


"""showDeckTransported()
showAgeWithTransported()
showBillWithTransported()
showBillWithCryo()
showVIPWithTransported()"""


# studyMissingValues(train)
separateColumns(train)
"""
createNoBill(train)

createLuxeBasic(train)
missingValuesCryoSleep(train)
missingValuesBill(train)
missingValueVIP(train)
"""
createSolo(train)
missingValueSide(train)
#missingValuesHomePlanet(train)
#studyMissingValuesSide(train)

"""
train_process = preprocessing(train)

#graph(train_process, "Transported", "l_needs")
test_process = preprocessing(test)
y = train["Transported"]

Logistic(train_process, test_process, y)"""