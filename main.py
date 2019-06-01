import pandas as pd
import keras

from data import data, weekday, encode_weekday

def main():
    d = data()
    print(weekday(d["date"][1]))
    print(encode_weekday(weekday(d["date"][1])))

if __name__ == "__main__":
    main()

