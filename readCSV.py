import pandas as pd


def readAndCombineCsv(**args):
    dfList = []
    for filename in args[1:]:
        df = pd.read_csv(filename)
        dfList.append(df)
    full = pd.concat(dfList)
    full.to_csv(args[0], encoding="utf-8")

