import os
import numpy as np

from functools import partial
from PIL import Image
from multiprocessing import Pool

from data import create_sales_csv, shop_sales, dataset, get_shops, get_items
from helpers import enqueue, mapnp, identity, arrayl, pmap, compose


def create_directories(names):
    """
    Creates a folder for each name in names
    """
    os.mkdir("./imageset")
    path = "./imageset/"

    for s in names:
        p = path + str(s)
        os.mkdir(p)
        os.mkdir(p + "/train")
        os.mkdir(p + "/test")


def date_loopup(item, data, day, month, year):
    if (month == 12):
        year = year - 1

    return data.loc[(data["day"] == day) &
                    (data["month"] == month) &
                    (data["year"] == year) &
                    (data["item_id"] == item)]["item_cnt_day"].sum()


def calendar_map(item, shop_data):
    item_results = []
    for year in (2013, 2014, 2015):
        for month in (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            monthly_sales = mapnp(lambda d: date_loopup(item, shop_data, d, month, year), range(1, 32))
            item_results = enqueue(item_results, monthly_sales)
    return item_results


def img(shop, item, folder, year, col):
    img = Image.fromarray(col)
    img.save("./imageset/" + str(shop) + "/" + folder + "/" + str(item) + "_" + str(year) + ".png")


def partition(sales_list):
    """
    Splits the item sales into annual sales + November sales
    """

    sales_nparr = mapnp(np.array, sales_list).astype(np.uint8)
    train = mapnp(lambda i: sales_nparr[12 * i: 12 * i + 11], range(3))
    test = mapnp(lambda i: [sales_nparr[12 * i + 11],], range(3))

    return [train, test]


def annotate_year(col):
    return mapnp(arrayl, zip(col, (2013, 2014, 2015)))


def image_item(shop, item):
    item_sales = partition(calendar_map(item, shop_sales(shop)))
    train, test = mapnp(annotate_year, item_sales)

    applied_img = partial(img, shop, item)
    mapnp(lambda t: applied_img("train", t[1], t[0]), train)
    mapnp(lambda t: applied_img("test", t[1], t[0]), test)


def generate_images():
    data = dataset()
    shops = get_shops(data)
    items = get_items(data)

    applied_image_item = map(lambda x: partial(image_item, x), shops)
    mapnp(lambda f: pmap(f, items), applied_image_item)


if __name__ == "__main__":
    create_directories(get_shops(dataset()))
    create_sales_csv()
    generate_images()

