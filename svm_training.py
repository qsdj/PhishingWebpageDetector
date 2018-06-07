from feature_engineering import *
import json
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from time import sleep, time
import numpy as np
import pandas as pd
import pickle
import random
import os

here = os.path.dirname(os.path.abspath(__file__))
# this code is based on data input of a list of lists which contain a group of URLs, n valid and others scam(
# the first, i.e. index 0 element is the home page)


"""make 4 panda dataframes for 4 different files"""
featuresDFLegit = pd.DataFrame(columns=['url', '2LD', 'SS', 'SS_url', 'i_url', 'i_2LD', 'title', 'language', 'xLink'])
featuresDFNotLegit = pd.DataFrame(
    columns=['url', '2LD', 'SS', 'SS_url', 'i_url', 'i_2LD', 'title', 'language', 'xLink'])
test_featuresDFLegit = pd.DataFrame(
    columns=['url', '2LD', 'SS', 'SS_url', 'i_url', 'i_2LD', 'title', 'language', 'xLink'])
test_featuresDFNotLegit = pd.DataFrame(
    columns=['url', '2LD', 'SS', 'SS_url', 'i_url', 'i_2LD', 'title', 'language', 'xLink'])


def parseData():
    """
    parse data from deltaphish_data.json into two list of dictionaries
    :return: a list of dictionary consisting of homePages and a list of dictionary consisting of pages
    """
    homePages = []
    with open('deltaphish_data.json', 'r') as fp:
        js = json.load(fp)
    indexList = []
    for i in range(len(js)):
        if js[i]['is_home']:
            homePages.append(js[i])
            indexList.append(i)
    indexList.sort(reverse=True)
    for j in indexList:
        js.pop(j)

    return js, homePages


def generateTrainTestData():
    """
    split data into training and testing set
    :return: trainingSet[], testSet[]
    """
    pages, homePages = parseData()
    groups = []
    testIndexes = []
    testSet = []
    for homePage in homePages:
        group = [homePage]
        for page in pages:
            if homePage['domain_name'] == page['domain_name']:
                group.append(page)
        groups.append(group)
    for i in range(int(len(groups) * 0.3)):
        rand = random.randint(0, len(groups) - 1)
        testIndexes.append(rand)
    testIndexes.sort(reverse=True)
    for i in testIndexes:
        testSet.append(groups.pop(i))
    trainingSet = groups
    return trainingSet, testSet


def featureExtraction_Engineering(trainingSet, testSet):
    """
    Extract all features from trainingSet and testSet, return them as list of lists, store them as csv at the same time
    :param trainingSet: a list of dictionaries representing homePages used to train
    :param testSet: a list of dictionaries representing homePages used to test
    :return:
        training data:
            not phishing
            phishing
        test data:
            not phishing
            phishing
    """

    global featuresDFLegit
    global featuresDFNotLegit
    global test_featuresDFNotLegit
    global test_featuresDFLegit

    finalDataLegit = []
    finalDataNotLegit = []
    testDataLegit = []
    testDataNotLegit = []
    for cluster in trainingSet:
        for i in range(1, len(cluster)):
            features = outputFeatures("HTML/" + str(cluster[0]['phishing_id']),
                                      "HTML/" + str(cluster[i]['phishing_id']))
            if features is not None:
                if cluster[i]['label'] == 0:
                    finalDataLegit.append(features)
                else:
                    finalDataNotLegit.append(features)
    for cluster in testSet:
        for i in range(len(cluster)):
            features = outputFeatures("HTML/" + str(cluster[0]['phishing_id']),
                                      "HTML/" + str(cluster[i]['phishing_id']))
            if features is not None:
                if cluster[i]['label'] == 0:
                    testDataLegit.append(features)
                else:
                    testDataNotLegit.append(features)

    trainGood = map(list, zip(*finalDataLegit))
    trainBad = map(list, zip(*finalDataNotLegit))
    testGood = map(list, zip(*testDataLegit))
    testBad = map(list, zip(*testDataNotLegit))

    test_featuresDFLegit = pd.DataFrame({'url': testGood[0], '2LD': testGood[1], 'SS': testGood[2],
                                         'SS_url': testGood[3], 'i_url': testGood[4],
                                         'i_2LD': testGood[5],
                                         'title': testGood[6], 'language': testGood[7],
                                         'xLink': testGood[8], 'isPhishing': 0},
                                        index=range(len(testGood[0])))

    test_featuresDFNotLegit = pd.DataFrame({'url': testBad[0], '2LD': testBad[1], 'SS': testBad[2],
                                            'SS_url': testBad[3], 'i_url': testBad[4],
                                            'i_2LD': testBad[5],
                                            'title': testBad[6], 'language': testBad[7],
                                            'xLink': testBad[8], 'isPhishing': 1},
                                           index=range(len(testBad[0])))

    featuresDFLegit = pd.DataFrame({'url': trainGood[0], '2LD': trainGood[1], 'SS': trainGood[2],
                                    'SS_url': trainGood[3], 'i_url': trainGood[4],
                                    'i_2LD': trainGood[5],
                                    'title': trainGood[6], 'language': trainGood[7],
                                    'xLink': trainGood[8], 'isPhishing': 0},
                                   index=range(len(trainGood[0])))

    featuresDFNotLegit = pd.DataFrame({'url': trainBad[0], '2LD': trainBad[1], 'SS': trainBad[2],
                                       'SS_url': trainBad[3], 'i_url': trainBad[4],
                                       'i_2LD': trainBad[5],
                                       'title': trainBad[6], 'language': trainBad[7],
                                       'xLink': trainBad[8], 'isPhishing': 1},
                                      index=range(len(trainBad[0])))

    return finalDataLegit, finalDataNotLegit, testDataLegit, testDataNotLegit


# x and y are list of good pages and list of bad pages
def train(x, y):
    """
    train a model using linear SVM with default parameters
    :param x: list of
    :param y:
    :return:
    """
    clf = svm.LinearSVC(verbose=True)
    train_X_list = []
    train_Y_list = []

    for i in x:
        train_X_list.append(i)
        train_Y_list.append(0)
    for i in y:
        train_X_list.append(i)
        train_Y_list.append(1)

    trainX = np.array(train_X_list)
    trainY = np.array(train_Y_list)
    print "training starts"
    clf.fit(trainX, trainY)
    print "train complete"
    x0, x1, y0, y1 = [], [], [], []
    for i in x:
        x0.append(i[0])
        x1.append(i[1])
    for i in y:
        y0.append(i[0])
        y1.append(i[1])

    plt.scatter(x0, x1, marker='+')
    plt.scatter(y0, y1, marker='o')
    plt.title('SVM Decision Region Boundary', size=16)
    plt.show()
    return clf


def main():
    print "parse data complete"
    trainingSet, testSet = generateTrainTestData()
    good, bad, testGood, testBad = featureExtraction_Engineering(trainingSet, testSet)
    print "feature extraction and feature engineering complete"
    model = train(good, bad)
    print model
    print test(model, testGood, testBad)
    return model


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
    for sample in testGood.iterrows():
        index, sample = sample
        if model.predict(np.asarray(sample.tolist()[:9]).reshape(1, -1)) == 0:
            correct += 1
        predictions.append(model.predict(np.asarray(sample.tolist()[:9]).reshape(1, -1)))
        actuals.append(0)
    for sample in testBad.iterrows():
        index, sample = sample
        if model.predict(np.asarray(sample.tolist()[:9]).reshape(1, -1)) == 1:
            correct += 1
        predictions.append(model.predict(np.asarray(sample.tolist()[:9]).reshape(1, -1)))
        actuals.append(1)
    return float(correct) / float(total), metrics.confusion_matrix(predictions, actuals)


# my_model = main()
# with open(os.path.join(here, "state.pickle"), "wb") as f:
#     pickle.dump(my_model, f)

with open(os.path.join(here, "state.pickle"), "rb") as f:
    my_model = pickle.load(f)

testGood1 = pd.read_csv("testGood.csv")
testBad1 = pd.read_csv("testBad.csv")

print test(my_model, testGood1, testBad1)


# featuresDFLegit.to_csv("features_extracted.csv", encoding="utf-8")
# featuresDFNotLegit.to_csv("features_extracted_not.csv", encoding="utf-8")
# test_featuresDFLegit.to_csv("testGood.csv", encoding="utf-8")
# test_featuresDFNotLegit.to_csv("testBad.csv", encoding="utf-8")


# dfList = [featuresDFLegit, featuresDFNotLegit, test_featuresDFLegit, test_featuresDFNotLegit]
# full_features = pd.concat(dfList)
# beautifulSoupDF.to_csv("beautiful_soup_extracted.csv", encoding="utf-8")
# full_features.to_csv("all_features.csv", encoding="utf-8")
