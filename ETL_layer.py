import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


path = '.\\competitive-data-science-predict-future-sales\\'

def load_data():
    """
    Funtion loads all dataset, transform date column into 
    correct format and create columns month and year.
    """
    train = pd.read_csv(path + "sales_train.csv")
    categories = pd.read_csv(path + 'item_categories.csv')
    items = pd.read_csv(path + 'items.csv')
    shops = pd.read_csv(path + 'shops.csv')
    test = pd.read_csv(path + 'test.csv')

    train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year

    return train, categories, items, shops, test

def remove_shop_duplicates(shops, train, test):
    """
    Function replaces all found shop duplicates.
    """
    shop_duplicates = {
        0: 57,
        1: 58,
        10: 11,
        39: 40
    }

    train['shop_id'] = train['shop_id'].replace(shop_duplicates)
    test['shop_id'] = test['shop_id'].replace(shop_duplicates)
    unique_shops = shops['shop_id'].replace(shop_duplicates)

    return unique_shops, train, test

def clean_train(train, test):
    """
    Function removes negative prices, outiners and shops 
    which weren't included in test, merges duplicates.
    """
    train = train.groupby(['date', 'date_block_num', 'shop_id', 'item_id']).agg({
        'item_price': 'mean',
        'item_cnt_day': 'sum'
    }).reset_index()

    train = train[train['item_cnt_day'] < 1001]
    train = train[train['item_price'] > 0 and train['item_price'] < 100000]

    train = train[train['shop_id'].isin(test['shop_id'].unique())]
    
    return train

