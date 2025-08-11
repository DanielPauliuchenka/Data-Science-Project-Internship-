import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


path = '.\\competitive-data-science-predict-future-sales\\'

train = pd.read_csv(path + "sales_train.csv")
item_categories = pd.read_csv(path + 'item_categories.csv')
items = pd.read_csv(path + 'items.csv')
shops = pd.read_csv(path + 'shops.csv')
test = pd.read_csv(path + 'test.csv')
