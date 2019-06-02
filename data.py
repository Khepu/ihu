import pandas as pd
import numpy as np
import datetime
import calendar

from functools import reduce, lru_cache
from helpers import identity, append, mapl, mapt, add_key


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


def weekday(date):
    day = datetime.datetime.strptime(date, '%d.%m.%Y').weekday()
    return (calendar.day_name[day])


def cyclic_encoder(x, fmap, total_x):
    degrees = np.deg2rad(90 + (360 / total_x) * fmap(x))
    return (np.cos(degrees), np.sin(degrees))


def encode_weekday(weekday):
    days = lambda x: {"Sunday": 0,
                      "Monday": 1,
                      "Tuesday": 2,
                      "Wednesday": 3,
                      "Thursday": 4,
                      "Friday": 5,
                      "Saturday": 6}[x]
    return cyclic_encoder(weekday, days, 7)


def encode_month(month):
    return cyclic_encoder(month - 1, identity, 12)


def split_date(date):
    date_tuple = mapt(int, date.split("."))
    date_tuple = append(date_tuple, encode_month(date_tuple[1]))
    date_tuple = append(date_tuple, encode_weekday(weekday(date)))

    return date_tuple


def add_city_column(shops):
    shops["city"] = shops["shop_name"].apply(lambda s: s.split(" ")[0])
    #Grabbing unique cities
    cities_set = set(shops["shop_name"].apply(lambda s: s.split(" ")[0]))

    #Indexing
    cities_indexed = zip(cities, range(len(cities_set)))

    #Pushing to dictionary {"city" : index}
    cities_dict = reduce(lambda dic, c: add_key(dic, c[0], c[1]), cities_indexed, {})

    shops["city"] = shops["city"].apply(lambda c: cities_dict[c])
    return shops


@lru_cache(maxsize=None)
def dataset():
    #data is not printable due to russian characters
    items, categories, sales, shops = loadData()
    mergedData = mergeData(items,
                           categories,
                           sales,
                           add_city_column(shops))

    finalData = mergedData.drop(axis=1,
                                labels=["item_category_name",
                                        "item_name",
                                        "shop_name"])

    finalData["date"] = finalData["date"].apply(split_date)

    return finalData

