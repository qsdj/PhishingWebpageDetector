from bs4 import BeautifulSoup
import urllib2
import tinycss2
from html_features import *
import cssutils


myBS = AdvancedBS('HTML/16')
styleTag = myBS.getStyles()
# media = cssutils.css.CSSMediaRule(styleTag[0])

print styleTag
