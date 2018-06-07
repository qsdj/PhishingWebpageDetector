# similarity based feature engineering
from html_features import *
import wordninja
from time import time
import pandas as pd
sample2URL = "https://connect.ultipro.com/"

beautifulSoupDF = pd.DataFrame()


def removeIdentical(ls):
    """make a list non-repetitive"""
    return list(set(ls))


def jaccard(list1, list2):
    """compute the jaccard index of two lists"""
    if len(list1) + len(list2) == 0:
        return 1.0
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / float(union)


def xLink(url1, url2):
    """
    :param url1: 1st url
    :param url2: 2nd url
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


def outputFeatures(url1, url2):
    """
    Output a set of features from two Urls
    :param url1:
    :param url2:
    :return:
    """
    feature_vec = []
    fake_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        sampleABS = AdvancedBS(url1)
        # bsDF = pd.read_html(sampleABS.getBs())
        sample2ABS = AdvancedBS(url2)
        # bsDF2 = pd.read_html(sample2ABS.getBs())

        # beautifulSoupDF.append(bsDF)
        # beautifulSoupDF.append(bsDF2)
    except urllib2.HTTPError:
        return fake_vec
    except socket.error:
        # time.sleep(2)
        print "webPage blocked our request"
        # outputFeatures(url1, url2)
        return None

    urlSet = sampleABS.getUrlSet()
    urlSet2 = sample2ABS.getUrlSet()
    domainSet = list(set(batch2LD(urlSet)))
    domainSet2 = list(set(batch2LD(urlSet2)))

    styleSet = sampleABS.getStyles()
    styleSet2 = sample2ABS.getStyles()
    styleSheets = sampleABS.getStyleSheetUrl()
    styleSheets2 = sample2ABS.getStyleSheetUrl()

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

    feature_vec.append(jaccard(urlSet, urlSet2))  # jaccard of url set
    feature_vec.append(jaccard(domainSet, domainSet2))  # jaccard of 2LD set
    feature_vec.append(jaccard(styleSet, styleSet2))  # jaccard of <style> tags
    feature_vec.append(jaccard(styleSheets, styleSheets2))  # jaccard of external sheets
    feature_vec.append(jaccard(IUrl1, IUrl2))  # jaccard of image urls
    feature_vec.append(jaccard(IDomain, IDomain2))  # jaccard of image 2LDs
    feature_vec.append(jaccard(TBOW1, TBOW2))  # jaccard of titles(bag of words)
    feature_vec.append(jaccard(language, language2))  # jaccard of language(s)
    feature_vec.append(xLink(sampleABS, sample2ABS))  # xLink is 1 if two pages are linked, 0 otherwise

    return feature_vec
