from bs4 import BeautifulSoup
import tldextract
import urllib2
import re
import socket
import requests
import pyjq
from time import time
from datetime import datetime
import cssutils


def isURL(string):
    """
    Check to see if a string conforms to rules of being a url
    :param string: the url input
    :return: True if the input is truly a url, False Otherwise
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, string)


def removeIdentical(ls):
    """make a list non-repetitive"""
    return list(set(ls))


def batch2LD(ls):
    """
    Helper function, change from a list of urls to a list of domains
    :param ls: a list of urls
    :return: a list of domains
    """
    newList = []
    for i in range(len(ls)):
        extracted = tldextract.extract(ls[i])
        if extracted.domain != '':
            newList.append(extracted.domain)
    return newList


def batchTag2Text(ls):
    """
    Helper function, parses a beautiful soup html tag into only its text
    :param ls: a list of tags
    :return: a list of texts
    """
    for i in range(len(ls)):
        ls[i] = ls[i].text
    return ls


mediaRule = 0


class AdvancedBS:
    """
    A wrapper class for BeautifulSoup to make retrieving of html information more straightforward

    Attributes:
        bs: the beautifulSoup
        html: html as a string
        url: url of html, string
    """
    def __init__(self, url):
        self.url = url
        if isURL(url):
            try:
                response = requests.get(url, timeout=3)
                self.html = response.content
                self.bs = BeautifulSoup(self.html, "lxml")
                pass
            except requests.exceptions.ConnectionError:
                raise urllib2.HTTPError(msg="Website already shut down", url=url, code=404, fp=None, hdrs=None)
            except requests.exceptions.ReadTimeout:
                raise socket.error("Requests sent too fast")
        else:
            print "getting data locally"
            dt1 = time()
            with open(url, 'r') as mf:
                self.html = mf.read().replace('\n', '')
            dt2 = time()
            print "open file time: " + str(dt2*1000 - dt1*1000)
            dt3 = time()
            self.bs = BeautifulSoup(self.html, "lxml")
            dt4 = time()
            print "create beautiful soup time: " + str(dt4*1000 - dt3*1000)

    def getBs(self):
        return self.bs

    def getTitle(self):
        title = self.bs.find('title')
        if title is not None:
            return title.text
        else:
            return ''
        # try:
        #     anchor = self.bs.find_all('a')[0]
        # except IndexError:
        #     return "not a title"
        # try:
        #     print "title : " + (anchor.get('title'))
        # except TypeError:
        #     print "There is no title"
        #     return "blank"
        # return anchor.get('title')

    def getHyperLinks(self):
        return self.bs.find_all("a")

    def getNumLines(self):
        """not functional"""
        count = len(self.html.readlines())
        return count

    def getMetaTags(self):
        return self.bs.find_all("meta")

    # This is done by getting all tags with attributes visibility:hidden
    def getHidden(self, display=False):
        if display:
            keyword = "display:none"
        else:
            keyword = "visibility:hidden"

        tags = self.bs.find_all()
        hiddenTextList = []
        for tag in tags:
            if 'style' in tag.attrs and keyword in tag.attrs['style']:
                hiddenTextList.append(tag)
        return hiddenTextList

    def getHTML(self):
        return self.html

    def getTextInBody(self):
        bodyTexts = []
        for tag in self.bs.find_all("body"):
            bodyTexts.append(tag.text)
        return bodyTexts

    def getLength(self):
        return len(self.html)

    # Not a definitive answer, only based on spaces
    def getAvgLengthOfWords(self):
        totalNumWords = 0
        totalLengthWords = 0
        for tag in self.bs.find_all():
            for word in tag.text.split(" "):
                totalLengthWords += len(word)
                totalNumWords += 1
        return totalLengthWords/totalNumWords

    def getWordCount(self):
        totalNumWords = 0
        for tag in self.bs.find_all():
            for word in tag.text.split(" "):
                totalNumWords += 1
        return totalNumWords

    def getDistinctWordCount(self):
        totalNumWords = 0
        wordList = []
        for tag in self.bs.find_all():
            for word in tag.text.split(" "):
                if word not in wordList:
                    totalNumWords += 1
                    wordList.append(word)
        return totalNumWords

    def getHiddenObjects(self):
        return self.getHidden(display=True)

    def getIFrames(self):
        return self.bs.find_all("iframe")

    def getLinksToScripts(self):
        linkList = []
        for tag in self.bs.find_all("script"):
            if "src" in tag.attrs:
                linkList.append(tag)
        return linkList

    # set url = false to get a list of all local paths
    def getImageSources(self, url=True):
        urlList = []
        for tag in self.bs.find_all("img"):
            # if "src" in tag.attrs and tag.attrs["src"] not in urlList or pathList:
            # if isURL(tag.attrs["src"]):
            try:
                urlList.append(tag.attrs["src"])
            except KeyError:
                pass
            # else:
            #    pathList.append(tag.attrs["src"])
        return urlList  # if url else pathList

    def getUrlSet(self):

        urlList = []
        for tag in self.bs.find_all("a"):
            if "href" in tag.attrs and tag.attrs["href"] not in urlList:
                if isURL(tag.attrs["href"]):
                    urlList.append(tag.attrs["href"])
        return urlList

    def getStyles(self):
        """not fully functional"""
        def parseSheet(sheetLocal):
            styleDictLocal = {}
            for rule in sheetLocal:
                selector = rule.selectorText
                styles = rule.style.cssText
                styleDictLocal[selector.encode('ascii', 'replace')] = styles.encode('ascii', 'replace')
            return styleDictLocal

        def expandDict(Dict):
            for key in Dict:
                newList = Dict[key].split(";\n")
                newDict = {}
                for j in newList:
                    newKey, value = j.split(": ")
                    newDict[newKey] = value
                Dict[key] = newDict
            return Dict

        styleList = []
        for tag in self.bs.find_all('style'):
            styleList.append(tag.text)
        for i in range(len(styleList)):
            sheet = cssutils.parseString(styleList[i])
            print sheet
            print self.url
            try:
                unFormatted = parseSheet(sheet)
            except AttributeError:
                mediaRule += 1
            styleDict = expandDict(unFormatted)
            styleList[i] = styleDict
        return styleList

    def getStylesSimple(self):
        styleList = []
        for tag in self.bs.find_all('style'):
            styleList.append(tag.text)
        return styleList

    def getStyleSheetUrl(self):
        urlList = []
        for tag in self.bs.find_all("link"):
            if "href" in tag.attrs and tag.attrs["href"] not in urlList and 'stylesheet' in tag.attrs['rel']:
                urlList.append(tag.attrs["href"])
        return removeIdentical(urlList)

    def getLanguageSet(self):
        languageList = []
        for tag in self.bs.find_all():
            if "lang" in tag.attrs and tag.attrs["lang"] not in languageList:
                languageList.append(tag.attrs["lang"])
        return languageList

