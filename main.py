import pandas as pd
import keras

from data import loadData, mergeData

def main():
    items, categories, sales, shops = loadData()
    mergeData(items, categories, sales, shops)

if __name__ == "__main__":
    main()

