from bs4 import BeautifulSoup
import urllib2


url = "https://www.ultimatesoftware.com/"
html = urllib2.urlopen(url).read()
soup = BeautifulSoup(html, "lxml")
tag = soup.find_all("a")
# print tag




