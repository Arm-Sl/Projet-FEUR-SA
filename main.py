import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

    while(len(currentInterval) > 0  or currentEmptySize < 10):
        currentAge += 1
        currentInterval = train.loc[(train["Age"] >= prevAge) & (train["Age"] < currentAge)]
        if(len(currentInterval) == 0):
            currentEmptySize += 1
            continue
        else:
            currentEmptySize = 0
        cor = abs(currentInterval["Age"].corr(currentInterval["Transported"]))

        if(prevCorr - cor >= 0.025 or prevCorr == cor):
            print("Change interval   :  pa " + str(prevAge)  + " ca " + str(currentAge) + "     " + str(prevCorr - cor) + "   " + str(prevCorr) + "   " + str(cor))
            liste.append(currentAge-1)
            prevAge = currentAge-1
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
        train.loc[(train['Age'] >= prevAge) & (train['Age'] <= liste[i]), "Age_group"] = i-1
        prevAge = liste[i]

    print(train["Age_group"].corr(train["Transported"]))


# Graphiques
def graph(data, x, y, type="strip"):
    if type == "strip":
        sns.stripplot(data=data, x=x, y=y, linewidth=0.1,s=1)
    if type == "count":
        data["t"] = data[x] + 2*data[y]
        sns.countplot(data=data, x="t")
        data.drop("t", axis=1, inplace=True)

    plt.show()

def showVIPWithTransported():
    sns.countplot(data=train, x="VIP", hue="Transported")
    plt.show()

def showBillWithCryo():
    fig = plt.figure(figsize=(10,20))
    bill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for i, name in enumerate(bill):
        ax = fig.add_subplot(5, 2, 2*i+1)
        sns.barplot(data=train, x="CryoSleep", ax=ax, y=name, errwidth=0)
        ax.set_title(name)
    fig.tight_layout()
    plt.show()

def showBillWithTransported():
    bill = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    fig = plt.figure(figsize=(10,20))
    for i, name in enumerate(bill):
        ax = fig.add_subplot(5, 2, 2*i+1)
        sns.histplot(data=train, x=name, axes=ax, bins=30, kde=False,  hue="Transported")
        ax.set_title(name)
    fig.tight_layout()
    plt.show()

def showDeckTransported():
    trainC = train.copy()
    trainC[["Deck", "Num", "Side"]] = trainC["Cabin"].str.split('/', expand=True)
    trainC.drop("Cabin", axis=1, inplace=True)
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(5,2,1)
    sns.countplot(data=trainC, x="Side", ax=ax, hue="Transported")

    ax =fig.add_subplot(5,2,3)
    sns.countplot(data=trainC, x="Deck", ax=ax, hue="Transported", order=["A","B","C", "D", "E", "F", "G", "T"])
    fig.tight_layout()
    plt.show()

def showAgeWithTransported():
    plt.figure(figsize=(10,4))
    sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
    plt.xlabel('Age')
    plt.show()

# Preprocessing

def preprocessing(df):
    #Split la colonne cabin en les colonnes Deck, Num et Side
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split('/', expand=True)

    #planetes
    homePlanete = pd.get_dummies(df["HomePlanet"])
    df.drop("HomePlanet", axis=1, inplace=True)

    sides = pd.get_dummies(df["Side"])
    df.drop("Side", axis=1, inplace=True)

    destination = pd.get_dummies(df["Destination"])
    df.drop("Destination", axis=1, inplace=True)

    df.drop("Cabin", axis=1, inplace=True)
    #Split la colonne PassengerId en les colonnes Group et NbInGroup
    df[["Group", "NbInGroup"]] = df["PassengerId"].str.split('_', expand=True)

    #Initialise les colones numeriques et nominales
    numerical = df.select_dtypes("float64",None)
    nominal = df.select_dtypes("object", None)

    nominal.drop("PassengerId", axis=1, inplace=True)
    nominal.drop("Name", axis=1, inplace=True)
    nominal.drop("NbInGroup", axis=1, inplace=True)


    # On remplace les données manquantes numériques
    numerical["Age"] = SimpleImputer(strategy="median").fit_transform(numerical[["Age"]])
    numerical["RoomService"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["RoomService"]])
    numerical["FoodCourt"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["FoodCourt"]])
    numerical["ShoppingMall"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["ShoppingMall"]])
    numerical["Spa"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["Spa"]])
    numerical["VRDeck"] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(numerical[["VRDeck"]])

    # Creation des tranches d'ages en fonction du graphe obtenu
    """numerical["Age_group"] = np.nan
    numerical.loc[(numerical['Age'] >= 0) & (numerical['Age'] <= 5), "Age_group"] = 0
    numerical.loc[(numerical['Age'] >= 6) & (numerical['Age'] <= 12), "Age_group"] = 1
    numerical.loc[(numerical['Age'] >= 13) & (numerical['Age'] <= 18), "Age_group"] = 2
    numerical.loc[(numerical['Age'] >= 19) & (numerical['Age'] <= 60), "Age_group"] = 3
    numerical.loc[(numerical['Age'] >= 61), "Age_group"] = 4"""
    findAgeIntervals(train)

    numerical.drop("Age", axis=1, inplace=True)

    numerical["b_needs"] = numerical["FoodCourt"] + numerical["ShoppingMall"]
    numerical["l_needs"] = numerical["RoomService"] + numerical["Spa"]+ numerical["VRDeck"]

    numerical.drop("FoodCourt", axis=1, inplace=True)
    numerical.drop("ShoppingMall", axis=1, inplace=True)
    numerical.drop("VRDeck", axis=1, inplace=True)
    numerical.drop("RoomService", axis=1, inplace=True)
    numerical.drop("Spa", axis=1, inplace=True)

   #nb de valeurs
   # print(train.nunique())

    # Remplace les strings par des int
    for column in nominal:
      nominal[column], uniques = pd.factorize(nominal[column])

    # Remplace les données manquantes
    for column in nominal:
      nominal[column] = SimpleImputer(missing_values=-1, strategy="most_frequent").fit_transform(nominal[[column]])

    newDf =  pd.concat([numerical, nominal, homePlanete, destination, sides], axis=1)
    # Normalise les données
    for column in newDf:
        newDf[column] = MinMaxScaler().fit_transform(np.array(newDf[column]).reshape(-1, 1))
    return newDf

# Random forest feature importante
def randomForest():
    print("feur")

def SVM(train_process, test_process, y):
    # Spécification des paramètres de la grille de recherche
    parameters = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01]
    }

    # Recherche des meilleurs paramètres
    search = GridSearchCV(SVC(), parameters, refit = True, verbose = 3).fit(train_process, y)


    # Entraînement du modèle avec les meilleurs paramètres
    svc_best = SVC(**search.best_params_)
    svc_best.fit(train_process, y)

    # Prédiction des étiquettes pour les données de test
    pred_train = svc_best.predict(test_process)
    submit = pd.DataFrame({'PassengerId': test["PassengerId"], 'Transported': pred_train})
    submit.to_csv("./Data/submit.csv", index=False)

def Logistic(train_process, test_process, y):
    parameters = {'penalty':['l1', 'l2'],
              'C': [0.1, 1, 10, 100],
              'solver':['saga', 'liblinear'],
              'max_iter':[100,1000,10000,100000]
              }
    Search = GridSearchCV(LogisticRegression(), parameters, scoring= 'accuracy', n_jobs=-1,cv=5,verbose=1).fit(train_process,y)

    log_model = LogisticRegression(penalty=Search.best_params_["penalty"],C=Search.best_params_["C"],max_iter=Search.best_params_["max_iter"],solver=Search.best_params_["solver"])
    log_model.fit(train_process,y)
    pred_train= log_model.predict(test_process)

    submit = pd.DataFrame({'PassengerId': test["PassengerId"], 'Transported': pred_train})
    submit.to_csv("./Data/submit.csv", index=False)

"""showDeckTransported()
showAgeWithTransported()
showBillWithTransported()
showBillWithCryo()
showVIPWithTransported()"""
train_process = preprocessing(train)

#graph(train_process, "Transported", "l_needs")
test_process = preprocessing(test)
y = train["Transported"]

Logistic(train_process, test_process, y)