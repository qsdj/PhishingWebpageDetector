# similarity based feature engineering
from html_features import AdvancedBS, batch2LD, removeIdentical
import wordninja
import pandas as pd
import pyxdameraulevenshtein as dl
from time import time
sample2URL = "https://connect.ultipro.com/"
beautifulSoupDF = pd.DataFrame()
import editdistance
SoupList = []


def AdvancedStrCompare(str1, str2, threshold=0.05):
    """the smaller threshold is, the less likely two strings will be considered same"""
    return 1 if dl.normalized_damerau_levenshtein_distance(str1, str2) < threshold else 0


# def LongStrAdvancedCompare(str1, str2):


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
    return float(intersection) / float(union) if union != 0 else 1


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

    if url1['url'] in url2['urlSet'] or url2['url'] in url1['urlSet']:
        return 1
    else:
        return 0


def outputFeatures(page, homePage):
    """
    Output a set of comparison based features from two dictionaries containing home pages and non-home pages
    input is from the output files of storeSoups()
    :param page: a non-home page
    :param homePage: its corresponding home page
    :return: Write all features to a csv file, return the features for each page as well.
    """
    dt1 = time()*1000
    feature_vec = []

    Title = page['title']
    Title2 = homePage['title']

    # print Title2
    TBOW1 = wordninja.split(Title.lower())
    TBOW2 = wordninja.split(Title2.lower())
    dt3 = time()*1000
    print "bag of words: " + str(dt3 - dt1)
    feature_vec.append(jaccard(page['urlSet'], homePage['urlSet']))  # jaccard of url set CHECKED!!!!
    print page['domain_name'], "home  ", homePage['domain_name']
    feature_vec.append(jaccard(page['domain_name'], homePage['domain_name']))  # jaccard of 2LD set CHECKED!!!!
    dt4 = time() * 1000
    # feature_vec.append(jaccard(page['styles'], homePage['styles'], advanced=True))
    #  jaccard of <style> tags HALF CHECKED!!!!
    list1 = page['styles']
    list2 = homePage['styles']

    # sameStyle = 0
    # for i in list1:
    #     for j in list2:
    #         if dictJaccard(list1, list2) > 0.7:
    #             if 'tagged' not in i or j:
    #                 sameStyle += 1
    #                 i['tagged'] = 1
    #                 j['tagged'] = 1
    #
    # feature_vec.append(float(sameStyle)/float(len(list1) + len(list2)))
    feature_vec.append(jaccard(list1, list2))
    dt7 = time() * 1000
    print "advanced style compare: " + str(dt7 - dt4)
    feature_vec.append(jaccard(page['styleSheetUrls'], homePage['styleSheetUrls']))
    # jaccard of external sheets HALF CHECKED!!!!!

    feature_vec.append(jaccard(removeIdentical(batch2LD(page['styleSheetUrls'])), removeIdentical(batch2LD(homePage['styleSheetUrls']))))

    feature_vec.append(jaccard(page['imageSources'], homePage['imageSources']))  # jaccard of image urls CHECKED!!!!

    print "first half: " + str(dt4 - dt3)
    feature_vec.append(jaccard(removeIdentical(batch2LD(page['imageSources'])), removeIdentical(
        batch2LD(page['imageSources']))))  # jaccard of image 2LDs CHECKED!!!!
    feature_vec.append(jaccard(TBOW1, TBOW2))  # jaccard of titles(bag of words) CHECKED!!!!
    feature_vec.append(jaccard(page['languages'], homePage['languages']))  # jaccard of language(s)
    feature_vec.append(xLink(page, homePage))  # xLink is 1 if two pages are linked, 0 otherwise
    dt2 = time()*1000

    print "feature output time: " + str(dt2-dt1)
    return feature_vec
