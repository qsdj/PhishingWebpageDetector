# similarity based feature engineering
from html_features import AdvancedBS, batch2LD, removeIdentical
import wordninja
import pandas as pd
import pyxdameraulevenshtein as dl
from time import time
sample2URL = "https://connect.ultipro.com/"
beautifulSoupDF = pd.DataFrame()


def AdvancedStrCompare(str1, str2, threshold=0.3):
    """the smaller threshold is, the less likely two strings will be considered same"""
    return 1 if dl.normalized_damerau_levenshtein_distance(str1, str2) <= threshold else 0


def jaccard(list1, list2, advanced=False):
    """compute the jaccard index of two lists"""
    intersection = 0
    if len(list1) + len(list2) == 0:
        return 1.0
    if advanced:
        for string1 in list1:
            for string2 in list2:
                if AdvancedStrCompare(string1, string2) == 1:
                    intersection += 1
                    continue
    else:
        intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / float(union)


def dictJaccard(dict1, dict2):
    """compute the jaccard index of two dictionaries"""
    intersection = 0
    union = sum(len(dict1[v]) for v in dict1) + sum(len(dict2[w]) for w in dict2)
    for key in dict1:
        if key in dict2:
            for innerKey in dict1[key]:
                if innerKey in dict2[key]:
                    if dict1[key][innerKey] == dict2[key][innerKey]:
                        intersection += 1
                        union -= 1
    if union == 0:
        return 1
    else:
        return float(intersection) / float(union)


def xLink(url1, url2):
    """
    :param url1: 1st advancedBS
    :param url2: 2nd advancedBS
    :return: 1 if two urls are linked, 0 otherwise
    """
    if type(url1) == type(url2) == str:
        abs1 = AdvancedBS(url1)
        abs2 = AdvancedBS(url2)
        if abs2 in abs1.getUrlSet() or abs1 in abs2.getUrlSet():
            return 1
        else:
            return 0

    if url1.url in url2.getUrlSet() or url2.url in url1.getUrlSet():
        return 1
    else:
        return 0


def outputFeatures(path1, path2, isPath=True):
    """
    Output a set of features from two Urls
    :param isPath: this means that the path1 and path2 parameters are indeed file paths
    :param path1:
    :param path2:
    :return:
    """
    dt1 = time() * 1000
    feature_vec = []
    if isPath:
        sampleABS = AdvancedBS(path1)
        # bsDF = pd.read_html(sampleABS.getBs())
        sample2ABS = AdvancedBS(path2)
        # bsDF2 = pd.read_html(sample2ABS.getBs())

        # beautifulSoupDF.append(bsDF)
        # beautifulSoupDF.append(bsDF2)
    else:
        dt7 = time() * 1000
        sampleABS = path1
        sample2ABS = path2
        print "what the: " + str(time()*1000 - dt7)
    dt2 = time() * 1000
    print "initialisation time: " + str(dt2-dt1)
    urlSet = sampleABS.getUrlSet()
    urlSet2 = sample2ABS.getUrlSet()
    domainSet = list(set(batch2LD(urlSet)))
    domainSet2 = list(set(batch2LD(urlSet2)))

    styleSet = sampleABS.getStylesSimple()
    styleSet2 = sample2ABS.getStylesSimple()
    styleSheets = sampleABS.getStyleSheetUrl()
    styleSheets2 = sample2ABS.getStyleSheetUrl()

    dt3 = time()*1000
    print "first part: " + str(dt3-dt2)

    IUrl1 = sampleABS.getImageSources()
    IUrl2 = sample2ABS.getImageSources()
    IDomain = removeIdentical(batch2LD(IUrl1))
    IDomain2 = removeIdentical(batch2LD(IUrl2))
    Title = sampleABS.getTitle()
    Title2 = sample2ABS.getTitle()
    TBOW1 = wordninja.split(Title.lower())
    TBOW2 = wordninja.split(Title2.lower())
    language = sampleABS.getLanguageSet()
    language2 = sampleABS.getLanguageSet()

    dt4 = time()*1000
    print "second part: " + str(dt4-dt3)
    feature_vec.append(jaccard(urlSet, urlSet2))  # jaccard of url set CHECKED!!!!
    feature_vec.append(jaccard(domainSet, domainSet2))  # jaccard of 2LD set CHECKED!!!!
    feature_vec.append(jaccard(styleSet, styleSet2, advanced=True))  # jaccard of <style> tags HALF CHECKED!!!!
    feature_vec.append(jaccard(styleSheets, styleSheets2))  # jaccard of external sheets HALF CHECKED!!!!!
    feature_vec.append(jaccard(removeIdentical(batch2LD(styleSheets)), removeIdentical(batch2LD(styleSheets2))))
    feature_vec.append(jaccard(IUrl1, IUrl2))  # jaccard of image urls CHECKED!!!!
    feature_vec.append(jaccard(IDomain, IDomain2))  # jaccard of image 2LDs CHECKED!!!!
    feature_vec.append(jaccard(TBOW1, TBOW2))  # jaccard of titles(bag of words) CHECKED!!!!
    feature_vec.append(jaccard(language, language2))  # jaccard of language(s)
    feature_vec.append(xLink(sampleABS, sample2ABS))  # xLink is 1 if two pages are linked, 0 otherwise

    return feature_vec
