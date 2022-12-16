import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler




# Null Values
#------------------------------------------------------------------------------------------------------------------------------#

def replace_null_with_mean(df,col):
    df[col]=df[col].replace(np.NaN,df[col].mean())
    return df

def replace_null_with_U(df,col):
    if col == 'PassengerId':
        df[col]=df[col].replace(np.NaN,"UUUU_UU")
    if col=='Cabin':
        df[col]=df[col].replace(np.NaN,"U/U/U")
    else:
        df[col]=df[col].replace(np.NaN,"U")
    return df

def replace_nulls_with_means(df):
    for col in ['Age','RoomService','FoodCourt','ShoppingMall','VRDeck','Spa']:
        df = replace_null_with_mean(df,col)
    return df

def replace_nulls_with_Us(df):
    for col in ['Cabin','HomePlanet','Destination','CryoSleep','VIP']:
        df = replace_null_with_U(df,col)
    return df





# String Formatting
#------------------------------------------------------------------------------------------------------------------------------#



"""
Basic string processing functions to use for 'Cabin' column.
"""
def find_deck(string):
    return string[0]
def find_number(string):
    m = re.search('/(.+?)/', string)
    return m.group(1)
def find_side(string):
    return string[-1]


def split_cabin_column(df):
    """
    Split 'Cabin' column into 'deck','number','side'
    """
   
    df['Cabin'] = df['Cabin'].astype(str)
    df['deck']=df['Cabin'].apply(find_deck)
    df['number']=df['Cabin'].apply(find_number)
    df['side']=df['Cabin'].apply(find_side)
    return df.drop(columns=['Cabin'])


def get_gggg(string):
    return string[:4]
def get_pp(string):
    return string[-2:]




def split_passenger_id_column(df):
    df['gggg'] = df['PassengerId'].apply(get_gggg)
    df['pp'] = df['PassengerId'].apply(get_pp)
    return df



# Merging HomePlanets for Groups
#------------------------------------------------------------------------------------------------------------------------------#
def update_homeplanet(gggg, dictionary_homeplanets):
    if gggg in dictionary_homeplanets.keys():
        return dictionary_homeplanets[gggg]
    else:
        return "U"

    
    
# OneHotEncoding
#------------------------------------------------------------------------------------------------------------------------------#
def one_hot_encode_columns(df,cols):
    for i in cols:
        x = pd.get_dummies(df[i], prefix=i)
        df = df.join(x)
        df = df.drop(columns=[i])
    return df




# ScaleFeatures
#------------------------------------------------------------------------------------------------------------------------------#
def scale_numerical_columns(df,cols, scaler=False):
    
    scaled_features = df.copy()
    features = scaled_features[cols]

    # Use scaler of choice; here Standard scaler is used
    if scaler == False:
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
    else:
        features = scaler.transform(features.values)

    scaled_features[cols] = features
    return scaled_features, scaler


