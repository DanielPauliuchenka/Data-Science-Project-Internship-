import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os


path = os.path.join('.', 'competitive-data-science-predict-future-sales')

def load_data():
    """
    Function loads all dataset, transforms date column into 
    correct format and creates columns month and year.
    """
    train = pd.read_csv(os.path.join(path, "sales_train.csv"))
    categories = pd.read_csv(os.path.join(path, 'item_categories.csv'))
    items = pd.read_csv(os.path.join(path, 'items.csv'))
    shops = pd.read_csv(os.path.join(path, 'shops.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))

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
    shops['shop_id'] = shops['shop_id'].replace(shop_duplicates)

    return shops, train, test

def clean_train(train, test):
    """
    Function removes negative prices, outliers and shops 
    which weren't included in test.
    """
    train = train.groupby(['date', 'date_block_num', 'shop_id', 'item_id']).agg({
        'item_price': 'mean',
        'item_cnt_day': 'sum'
    }).reset_index()

    train = train[train['item_cnt_day'] < 1001]
    train = train[train['item_price'] > 0]
    train = train[train['item_price'] < 100000]
    train = train[train['shop_id'].isin(test['shop_id'].unique())]
    
    return train

def month_aggregate(train, items, test=None, is_test=False):
    """
    Function groups sales data by month and aggregates it into 
    new data such as item_cnt_month. Also it makes a grid and fills months 
    without sales with zeros to provide a model with data on months when there
    were no sales. Also, it prepares the dataset for testing a model.
    """
    monthly = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
        'item_price': 'mean',
        'item_cnt_day': 'sum'
    }).rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'mean_price'}).reset_index()

    all_shops = monthly['shop_id'].unique()
    all_items = monthly['item_id'].unique()
    all_blocks = range(0, 35) if is_test else range(0, 34)

    grid = pd.DataFrame(list(product(all_blocks, all_shops, all_items)), 
                        columns=['date_block_num', 'shop_id', 'item_id'])

    grid = grid.merge(monthly, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)
    grid['mean_price'] = grid['mean_price'].fillna(grid['mean_price'].mean())

    grid['item_cnt_month'] = np.clip(grid['item_cnt_month'], 0, 20)

    grid = grid.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

    if is_test:
        test = test.copy()
        test['date_block_num'] = 34
        grid = test[['ID', 'shop_id', 'item_id', 'date_block_num']] \
               .merge(grid, on=['date_block_num', 'shop_id', 'item_id'], how='left')

        grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)
        grid['mean_price'] = grid['mean_price'].fillna(grid['mean_price'].mean())

    return grid

def add_lags_and_rollings(train):
    """
    Function generates lags and rolling mean.
    """
    train = train.sort_values(['shop_id', 'item_id', 'date_block_num'])

    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        train[f'item_cnt_lag_{lag}'] = train.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)

    train['rolling_mean'] = train.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(3).mean())

    lag_cols = [col for col in train.columns if 'lag' in col or 'rolling' in col]
    train[lag_cols] = train[lag_cols].fillna(0)
    
    return train

def etl_process():
    train, categories, items, shops, test = load_data()
    print("Data loaded successfully!")

    shops, train, test = remove_shop_duplicates(shops, train, test)
    print("Shop duplicates are removed!")

    train = clean_train(train, test)
    print("Train data is cleaned!")

    train_grid = month_aggregate(train, items)
    print("Train data is aggregated!")
    train_grid = add_lags_and_rollings(train_grid)
    print("Lags and rolling mean were generated!")

    test_grid = month_aggregate(train, items, test=test, is_test=True)
    print("Test data is aggregated!")
    test_grid = add_lags_and_rollings(test_grid)
    print("Lags and rolling mean were generated!")

    train_grid.to_csv('processed_train.csv', index=False)
    test_grid.to_csv('processed_test.csv', index=False)
    
    print("ETL completed. Files saved: processed_train.csv and processed_test.csv")

if __name__ == "__main__":
    etl_process()