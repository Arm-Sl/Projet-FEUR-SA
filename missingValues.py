__author__ = 'Antonin'
__Filename = 'missingValues'
__Creationdate__ = '27/03/2023'

import numpy as np
from pandas import DataFrame


def missingValuesBill(df: DataFrame):
    bill = np.array(["RoomService", "Spa", "ShoppingMall", "VRDeck", "FoodCourt"])
    print("Bill missing values :", df[bill].isna().sum().sum())
    for b in bill:
        df.loc[(df[b].isna()) & (df['CryoSleep'] is True), b] = 0
    print("Bill missing values :", df[bill].isna().sum().sum())

    for b in bill:
        df.loc[(df[b].isna()), b] = df[b].median()
    print("Bill missing values :", df[bill].isna().sum().sum())

    # Update NoBill Column
    df.loc[(df["RoomService"]+df["Spa"]+df["ShoppingMall"]+df["VRDeck"]+df["FoodCourt"] == 0), "NoBill"] = 1
    # Update Luxe/Basics
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["Basics"] = df["ShoppingMall"] + df["FoodCourt"]

def missingValuesHomePlanet(df: DataFrame):
    print("HomePlanet missing values :", len(df[df['HomePlanet'].isna()]))
    linesMissingHomePlanet = df[df['HomePlanet'].isna()]

    for index, row in linesMissingHomePlanet.iterrows():
        group = df.loc[df["Group"] == row["Group"], np.array(["HomePlanet", "Group"])]
        homeplanet = group[~group["HomePlanet"].isna()].index
        if len(homeplanet.values) > 0:
            df.at[index, "HomePlanet"] = group["HomePlanet"][homeplanet.values[0]]

    print("HomePlanet missing values :", len(df[df['HomePlanet'].isna()]))

    linesMissingHomePlanet = df[df['HomePlanet'].isna()]

    for index, row in linesMissingHomePlanet.iterrows():
        surname = df.loc[df["Surname"] == row["Surname"], np.array(["HomePlanet", "Surname"])]
        homeplanet = surname[~surname["HomePlanet"].isna()].index
        if len(homeplanet.values) > 0:
            if index == 5371:
                print(homeplanet.values)
            df.at[index, "HomePlanet"] = surname["HomePlanet"][homeplanet.values[0]]
    print("HomePlanet missing values :", len(df[df['HomePlanet'].isna()]))
    df.loc[(df['HomePlanet'].isna()) & ~(df['Deck']=='D'), 'HomePlanet']='Earth'
    df.loc[(df['HomePlanet'].isna()) & (df['Deck']=='D'), 'HomePlanet']='Mars'
    print("HomePlanet missing values :", len(df[df['HomePlanet'].isna()]))

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
        group = df.loc[(df["Group"] == row["Group"]), np.array(["Side","Group"])]
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

def missingValueDestination(df: DataFrame):
    missingDestination = df.loc[(df["Destination"].isna())]
    print("Destination missing values: ", df["Destination"].isna().sum())
    for index, row in missingDestination.iterrows():
        group = df.loc[(df["Group"] == row["Group"]), np.array(["Destination","Group"])]
        side = group[~group["Destination"].isna()].index
        if len(side.values) > 0:
            df.at[index, "Destination"] = group["Destination"][side.values[0]]
    print("Destination missing values: ", df["Destination"].isna().sum())
    df.loc[(df["Destination"].isna()), "Destination"] = "TRAPPIST-1e"
    print("Destination missing values: ", df["Destination"].isna().sum())

def missingValueDeck(df: DataFrame):
    missingDeck = df.loc[(df["Deck"].isna())]
    print("Deck missing values: ", df["Deck"].isna().sum())
    for index, row in missingDeck.iterrows():
        group = df.loc[(df["Group"] == row["Group"]), np.array(["Deck", "Group"])]
        side = group[~group["Deck"].isna()].index
        if len(side.values) > 0:
            df.at[index, "Deck"] = group["Deck"].value_counts().index[0]
    print("Deck missing values: ", df["Deck"].isna().sum())
    df.loc[(df["Deck"].isna()) & (df["HomePlanet"] == "Earth"), "Deck"] = "G"
    df.loc[(df["Deck"].isna()) & (df["HomePlanet"] == "Mars"), "Deck"] = "F"
    df.loc[(df["Deck"].isna()) & (df["HomePlanet"] == "Europa") & (df["Solo"] == 1), "Deck"] = "C"
    df.loc[(df["Deck"].isna()) & (df["HomePlanet"] == "Europa") & (df["Solo"] == 0), "Deck"] = "B"
    print("Deck missing values: ", df["Deck"].isna().sum())



def missingValues(df: DataFrame):
    missingValuesHomePlanet(df)
    missingValuesBill(df)
    missingValuesCryoSleep(df)
    missingValueDestination(df)
    missingValueSide(df)
    missingValueVIP(df)
    missingValueDeck(df)
