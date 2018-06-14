from html_features import *
from feature_engineering import *
from svm_training import *

testBS = AdvancedBS("HTML/test.html")
testBS2 = AdvancedBS("HTML/136")


def main_test(Abs):
    print "title: " + str(Abs.getTitle())
    print "set of url: " + str(Abs.getUrlSet())
    print "languages: " + str(Abs.getLanguageSet())
    print "styles: " + str(Abs.getStyles())
    print "image sources: " + str(Abs.getImageSources())
    print "external stylesheet sources: " + str(Abs.getStyleSheetUrl())
    print "set of domain: " + str(removeIdentical(batch2LD(Abs.getUrlSet())))
    print "image 2LD: " + str(removeIdentical(batch2LD(Abs.getImageSources())))
    print "style sheets 2LD: " + str(removeIdentical(batch2LD(Abs.getStyleSheetUrl())))


def store_TEST():
    js, homePages = parseData()
    storeSoups(js, homePages)



store_TEST()
# print outputFeatures(testBS, testBS2, isPath=False)
