import os
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

DataFrame = pd.DataFrame
Series = pd.Series


class ETLProcessor:
    """
    A class to perform a full ETL process on sales data.

    This class encapsulates the entire logic: from loading the raw data
    to cleaning, aggregating, enriching with new features (lags), and
    saving the ready-for-modeling datasets.

    Attributes:
        data_path (str): The path to the directory with the source CSV files.
        raw_train (Optional[DataFrame]): Raw sales data.
        raw_test (Optional[DataFrame]): Raw test data.
        raw_items (Optional[DataFrame]): Raw item data.
        raw_shops (Optional[DataFrame]): Raw shop data.
        raw_categories (Optional[DataFrame]): Raw category data.
        processed_train (Optional[DataFrame]): The processed training set.
        processed_test (Optional[DataFrame]): The processed test set.
    """

    _SHOP_DUPLICATES = {0: 57, 1: 58, 10: 11, 39: 40}
    _LAG_PERIODS = [1, 2, 3, 6, 12]

    def __init__(self, data_path: str):
        """
        Initializes the ETL processor.

        Args:
            data_path (str): The path to the folder containing files
                             like 'sales_train.csv', 'test.csv', etc.
        """
        self.data_path: str = data_path
        self.raw_train: Optional[DataFrame] = None
        self.raw_test: Optional[DataFrame] = None
        self.raw_items: Optional[DataFrame] = None
        self.raw_shops: Optional[DataFrame] = None
        self.raw_categories: Optional[DataFrame] = None

        self.processed_train: Optional[DataFrame] = None
        self.processed_test: Optional[DataFrame] = None

    def _load_data(self) -> None:
        """
        Loads all datasets from CSV files.

        The data is stored in the 'raw_*' attributes of the class instance.
        It also converts the 'date' column and creates 'month' and 'year'.
        """
        print("Step 1/5: Loading data...")
        self.raw_train = pd.read_csv(os.path.join(self.data_path, "sales_train.csv"))
        self.raw_test = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
        self.raw_items = pd.read_csv(os.path.join(self.data_path, 'items.csv'))
        self.raw_shops = pd.read_csv(os.path.join(self.data_path, 'shops.csv'))
        self.raw_categories = pd.read_csv(os.path.join(self.data_path, 'item_categories.csv'))
        
        self.raw_train['date'] = pd.to_datetime(self.raw_train['date'], format='%d.%m.%Y')

    def _preprocess_raw_data(self) -> None:
        """
        Performs preliminary cleaning of the raw data.

        - Removes shop duplicates.
        - Cleans the training data from outliers and anomalies.
        """
        print("Step 2/5: Preprocessing and cleaning data...")
        for df in [self.raw_train, self.raw_test, self.raw_shops]:
            df['shop_id'] = df['shop_id'].replace(self._SHOP_DUPLICATES)

        self.raw_train = self.raw_train[self.raw_train['item_price'] < 100000]
        self.raw_train = self.raw_train[self.raw_train['item_price'] > 0]
        self.raw_train = self.raw_train[self.raw_train['item_cnt_day'] < 1001]
        
        self.raw_train = self.raw_train[self.raw_train['shop_id'].isin(self.raw_test['shop_id'].unique())]


    def _create_master_grid(self) -> DataFrame:
        """
        Creates the main grid for all month-shop-item combinations.

        This is a key step to create a complete time series,
        filling in missing months with zero sales.

        Returns:
            DataFrame: A complete data grid with aggregated monthly sales.
        """
        print("Step 3/5: Aggregating data by month and creating the grid...")
        monthly_sales = self.raw_train.groupby(
            ['date_block_num', 'shop_id', 'item_id'], as_index=False
        ).agg(item_cnt_month=('item_cnt_day', 'sum'))

        grid = []
        for block_num in self.raw_train['date_block_num'].unique():
            shops_in_block = self.raw_train.loc[self.raw_train['date_block_num'] == block_num, 'shop_id'].unique()
            items_in_block = self.raw_train.loc[self.raw_train['date_block_num'] == block_num, 'item_id'].unique()
            grid.append(np.array(list(product([block_num], shops_in_block, items_in_block)), dtype='int32'))
        
        grid = pd.DataFrame(np.vstack(grid), columns=['date_block_num', 'shop_id', 'item_id'])
        
        grid = pd.merge(grid, monthly_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)
        
        self.raw_test['date_block_num'] = 34
        grid = pd.concat([grid, self.raw_test[['date_block_num', 'shop_id', 'item_id']]], ignore_index=True, sort=False)
        grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)

        grid['item_cnt_month'] = np.clip(grid['item_cnt_month'], 0, 20)

        return grid

    def _generate_features(self, master_grid: DataFrame) -> DataFrame:
        """
        Generates lag features for the time series.

        Args:
            master_grid (DataFrame): The data grid for which to create features.

        Returns:
            DataFrame: The data grid enriched with lag features.
        """
        print("Step 4/5: Generating lag features...")
        grid = master_grid.copy()
        
        grid = grid.sort_values(['shop_id', 'item_id', 'date_block_num'])
        
        for lag in self._LAG_PERIODS:
            shifted = grid.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
            grid[f'item_cnt_lag_{lag}'] = shifted

        grid = pd.merge(grid, self.raw_items[['item_id', 'item_category_id']], on='item_id', how='left')

        lag_cols = [col for col in grid.columns if 'lag' in col]
        grid[lag_cols] = grid[lag_cols].fillna(0)
        
        return grid

    def run(self) -> None:
        """
        Runs the full ETL pipeline.
        
        Sequentially executes all steps: loading, cleaning,
        aggregating, feature creation, and splitting into train/test sets.
        """
        self._load_data()
        self._preprocess_raw_data()
        master_grid = self._create_master_grid()
        featured_grid = self._generate_features(master_grid)
        
        self.processed_train = featured_grid[featured_grid['date_block_num'] < 34].copy()
        self.processed_test = featured_grid[featured_grid['date_block_num'] == 34].copy()
        
        self.processed_test = pd.merge(
            self.raw_test,
            self.processed_test,
            on=['date_block_num', 'shop_id', 'item_id'],
            how='left'
        )

        print("\nETL process completed successfully!")

    def save(self, output_dir: str = '.') -> None:
        """
        Saves the processed dataframes to CSV files.

        Args:
            output_dir (str, optional): The directory to save the files in.
                                        Defaults to the current directory.
        """
        if self.processed_train is None or self.processed_test is None:
            raise RuntimeError("Processing has not been run. Call the .run() method before saving.")
        
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'processed_train.csv')
        test_path = os.path.join(output_dir, 'processed_test.csv')

        self.processed_train.to_csv(train_path, index=False)
        self.processed_test.to_csv(test_path, index=False)
        print(f"\nResults saved to:\n- {train_path}\n- {test_path}")