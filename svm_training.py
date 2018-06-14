from feature_engineering import *
import sklearn
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from time import sleep, time
import numpy as np
import pickle
import os
from html_features import *
import itertools
here = os.path.dirname(os.path.abspath(__file__))


def generateTrainTestData(allData):
    """
    split data into training and testing set
    :return: trainingSet[], testSet[]
    """

    nonHomePages, homePages = allData.loc[allData['is_home'] == False], allData.loc[allData['is_home'] == True]
    nonHomePages.dropna(inplace=True)
    homePages.dropna(inplace=True)
    print 'dataFrame preprocessing done'
    phishList = []
    goodList = []
    idSet = list(set(nonHomePages['phishing_id']))
    for i in idSet:
        oneGroupOfNonHomePages = nonHomePages.loc[nonHomePages['phishing_id'] == i]
        homePage = homePages.loc[homePages['phishing_id'] == i]

        for row in oneGroupOfNonHomePages.T.to_dict().values():
            dt1 = time()*1000
            for r in homePage.T.to_dict().values():
                if row['label'] == 0:
                    goodList.append(outputFeatures(row, r))
                else:
                    phishList.append(outputFeatures(row, r))
            dt2 = time()*1000
            print dt2 - dt1
    print 'features ready'

    zeros = []
    for i in range(len(goodList)):
        zeros.append(0)
    ones = []
    for i in range(len(phishList)):
        ones.append(1)
    goodPageTrain, goodPageTest, labelTrain, labelTest = sklearn.model_selection.train_test_split(
        goodList, zeros, test_size=0.3)

    phishPageTrain, phishPageTest, phishLabelTrain, phishLabelTest = sklearn.model_selection.train_test_split(
        phishList, ones, test_size=0.3)

    createLocalCsv(phishPageTrain, goodPageTrain, "trainingFeatures")
    createLocalCsv(phishPageTest, goodPageTest, "testingFeatures")

    return [goodPageTrain, goodPageTest, labelTrain, labelTest, phishPageTrain, phishPageTest, phishLabelTrain,
            phishLabelTest]


def createLocalCsv(phishX, goodX, name):
    # store features in csv
    phishX = map(list, zip(*phishX))
    goodX = map(list, zip(*goodX))
    phishXCsv = pd.DataFrame({'url': phishX[0], '2LD': phishX[1], 'SS': phishX[2],
                              'SS_url': phishX[3], 'i_url': phishX[4],
                              'i_2LD': phishX[5],
                              'title': phishX[6], 'language': phishX[7],
                              'xLink': phishX[8], 'isPhishing': 1},
                             index=range(len(phishX[0])))
    goodXCsv = pd.DataFrame({'url': goodX[0], '2LD': goodX[1], 'SS': goodX[2],
                             'SS_url': goodX[3], 'i_url': goodX[4],
                             'i_2LD': goodX[5],
                             'title': goodX[6], 'language': goodX[7],
                             'xLink': goodX[8], 'isPhishing': 0},
                            index=range(len(goodX[0])))

    toCsv = pd.concat([phishXCsv, goodXCsv])
    toCsv.to_csv(name + ".csv", encoding='utf-8')


def train(phishX, goodX):
    """
    train a model using linear SVM with default parameters
    :param phishX: list of lists representing feature vectors of phishing pages
    :param goodX: list of lists representing feature vectors of safe pages
    :return: phishX and goodX (for later use), and the svm model trained
    """
    print 'features successfully stored'
    clf = svm.LinearSVC(C=0.5, verbose=True)

    train_X_list = []
    train_Y_list = []
    for i in phishX:
        train_X_list.append(i[:9])
        train_Y_list.append(1)
    for i in goodX:
        train_X_list.append(i[:9])
        train_Y_list.append(0)
    trainX = np.array(train_X_list)
    trainY = np.array(train_Y_list)

    print "training starts"
    clf.fit(trainX, trainY)
    print "train complete"

    return clf, phishX, goodX


def plot(phishX, goodX, pltDimensions=(0, 1)):
    """

    :param phishX: list of lists representing feature vectors of phishing pages
    :param goodX: list of lists representing feature vectors of safe pages
    :param pltDimensions:
    :return:
    """

    print 'plotting'
    x0, x1, y0, y1 = [], [], [], []
    for i in phishX:
        x0.append(i[pltDimensions[0]])
        x1.append(i[pltDimensions[1]])
    for i in goodX:
        y0.append(i[pltDimensions[0]])
        y1.append(i[pltDimensions[1]])

    plt.scatter(x0, x1, marker='+')
    plt.scatter(y0, y1, marker='o')
    plt.title('showing plots of parts of the vector', size=16)
    options = {0: plt.xlabel("url jaccard"),
               1: plt.xlabel("2LD jaccard"),
               2: plt.xlabel("SS jaccard"),
               3: plt.xlabel("SS-url jaccard"),
               4: plt.xlabel("SS-2LD jaccard"),
               5: plt.xlabel("Image-url jaccard"),
               6: plt.xlabel("Image-2LD jaccard"),
               7: plt.xlabel("title jaccard"),
               8: plt.xlabel("language jaccard"),
               9: plt.xlabel("Xlink"),
               10: plt.ylabel("url jaccard"),
               11: plt.ylabel("2LD jaccard"),
               12: plt.ylabel("SS jaccard"),
               13: plt.ylabel("SS-url jaccard"),
               14: plt.ylabel("SS-2LD jaccard"),
               15: plt.ylabel("Image-url jaccard"),
               16: plt.ylabel("Image-2LD jaccard"),
               17: plt.ylabel("title jaccard"),
               18: plt.ylabel("language jaccard"),
               19: plt.ylabel("Xlink")
               }
    print pltDimensions
    options[pltDimensions[0]]
    options[pltDimensions[1] + 10]
    # plt.show()


def plotAllCombinations(phishX, goodX, pltDimension=2):
    print 'plotting all combinations'
    dimension = len(phishX[0])
    if pltDimension > 3:
        raise ValueError("Can't visualize such high dimensional data")
    allCombinations = itertools.combinations(range(dimension), pltDimension)
    for i in allCombinations:
        plot(phishX, goodX, tuple(i))


def test(model, testGood, testBad):
    """
    test the input model and return:
        :CorrectRate
        :FalsePositiveRate
        :FalseNegativeRate
    """
    correct = 0
    predictions = []
    actuals = []
    total = len(testGood) + len(testBad)
    for sample in testGood:
        if model.predict(np.asarray(sample[:9]).reshape(1, -1)) == 0:
            correct += 1
        predictions.append(model.predict(np.asarray(sample[:9]).reshape(1, -1)))
        actuals.append(0)
    for sample in testBad:
        if model.predict(np.asarray(sample[:9]).reshape(1, -1)) == 1:
            correct += 1
        predictions.append(model.predict(np.asarray(sample[:9]).reshape(1, -1)))
        actuals.append(1)
    return float(correct) / float(total), metrics.confusion_matrix(predictions, actuals)


def main():
    full, good, phish = parseData()
    print "parse data complete"
    fullList = generateTrainTestData(full)
    print "feature extraction and feature engineering complete"
    model, phishX, goodX = train(fullList[4], fullList[0])
    plotAllCombinations(phishX, goodX, pltDimension=2)
    print test(model, fullList[1], fullList[5])
    with open(os.path.join(here, "state.pickle"), "wb") as f:
        pickle.dump(model, f)
    print 'model stored'


main()
# with open(os.path.join(here, "state.pickle"), "rb") as f:
#     my_model = pickle.load(f)

