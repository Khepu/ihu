import pandas as pd
import keras

from functools import reduce

def merge(left, right, left_label, right_label=False):
    return left.merge(right, "inner", left_on=left_label, right_on=(right_label if right_label else left_label))

def loadData():
    items = pd.read_csv("./dataset/items.csv")
    categories = pd.read_csv("./dataset/item_categories.csv")
    sales = pd.read_csv("./dataset/sales_train_v2.csv")
    shops = pd.read_csv("./dataset/shops.csv")

    return [items, categories, sales, shops]

def mergeData(items, categories, sales, shops):
    join_table_info = [[categories, "item_category_id"],
                       [sales, "item_id"],
                       [shops, "shop_id"]]

    return reduce(lambda acc, val: merge(acc, val[0], val[1]), join_table_info, items)

def main():
    items, categories, sales, shops = loadData()
    mergeData(items, categories, sales, shops)
    
if __name__ == "__main__":
    main()

