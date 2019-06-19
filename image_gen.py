import os
import numpy as np

from functools import partial
from PIL import Image
from multiprocessing import Pool

from data import create_sales_csv, shop_sales, dataset, get_shops, get_items
from helpers import enqueue, mapnp, identity, arrayl, pmap



def create_directories(names):
    path = "./imageset/"

    for s in names:
        p = path + str(s)
        os.mkdir(p)
        os.mkdir(p + "/train")
        os.mkdir(p + "/test")


def calendar_fn(item, data, day, month, year):
    # i -> [[a]] -> Integer -> Integer -> Integer -> Integer
    if (month == 12):
        year = year - 1

    return data.loc[(data["day"] == day) &
                    (data["month"] == month) &
                    (data["year"] == year) &
                    (data["item_id"] == item)]["item_cnt_day"].sum()


def calendar_map(f, col, shop_data):
    result = []
    for item in col:
        item_results = []
        for year in (2013, 2014, 2015):
            for month in (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
                monthly_sales = mapnp(lambda d: f(item, shop_data, d, month, year), range(1, 32))
                item_results = enqueue(item_results, monthly_sales)
        result = enqueue(result, item_results)
    return result


def img(shop, item, folder, year, col):
    img = Image.fromarray(col)
    img.save("./imageset/" + shop + "/" + folder + "/" + item + "_" + year + ".png")


def partition(sales_list):
    """
    Splits the item sales into annual sales + November sales
    """

    sales_nparr = mapnp(np.array, sales_list).astype(np.uint8)
    train = mapnp(lambda i: sales_nparr[12 * i: 12 * i + 11], range(3))
    test = mapnp(lambda i: sales_nparr[12 * i + 11], range(3))

    return [mapnp(arrayl, zip(train, (2013, 2014, 2015))),
            mapnp(arrayl, zip(test, (2013, 2014, 2015)))]


def generate_images():
    data = dataset()
    shops = get_shops(data)
    items = get_items(data)
    create_sales_csv()

    #create a folder for each shop
    create_directories(shops)

    sales = pmap(shop_sales, shops)
    dated_sales_per_shop = zip(pmap(partial(calendar_map, calendar_fn, items), sales), shops)

