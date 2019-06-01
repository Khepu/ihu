import pandas as pd
import numpy as np
import datetime
import calendar

from functools import reduce


def merge(left, right, left_label, right_label=False):
    return left.merge(right, "inner", left_on=left_label, right_on=(right_label if right_label else left_label))


def loadData():
    enc = "utf-8"
    items = pd.read_csv("./dataset/items.csv", encoding=enc)
    categories = pd.read_csv("./dataset/item_categories.csv", encoding=enc)
    sales = pd.read_csv("./dataset/sales_train_v2.csv", encoding=enc)
    shops = pd.read_csv("./dataset/shops.csv", encoding=enc)

    return [items, categories, sales, shops]


def mergeData(items, categories, sales, shops):
    join_table_info = [[categories, "item_category_id"],
                       [sales, "item_id"],
                       [shops, "shop_id"]]

    return reduce(lambda acc, val: merge(acc, val[0], val[1]), join_table_info, items)


def data():
    #data is not printable due to russian characters
    items, categories, sales, shops = loadData()
    mergedData = mergeData(items, categories, sales, shops)
    finalData = mergedData.drop(labels=["item_category_name",
                            "item_name",
                            "shop_name"],
                    axis=1)
    return finalData


def weekday(date):
    day = datetime.datetime.strptime(date, '%d.%m.%Y').weekday()
    return (calendar.day_name[day])


def encode_to_cycle(x, fmap):
    degrees = np.deg2rad(90 + 360 / fmap(x))
    return (np.cos(degrees), np.sin(degrees))


def encode_weekday(weekday):
    days = lambda x: {"Sunday": 1,
                      "Monday": 2,
                      "Tuesday": 3,
                      "Wednesday": 4,
                      "Thursday": 5,
                      "Friday": 6,
                      "Saturday": 7}[x]
    return encode_to_cycle(weekday, days)


def encode_months(month):
    encode_to_cycle(month, identity)


