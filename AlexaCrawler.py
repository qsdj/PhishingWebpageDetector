from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from urlparse import urlparse
import dns.resolver



DRIVER = '/Users/usgxchen/Downloads/chromedriver'
global_num = 0
URL = 'https://www.alexa.com/siteinfo'



def crawlRanking(url):
    driver = webdriver.Chrome(DRIVER)
    driver.get(url)
    inputElement = driver.find_element_by_id('siteInput')
    inputElement.send_keys(url)
    inputElement.send_keys(Keys.ENTER)
    time.sleep(1)
    outputElement = driver.find_element_by_class_name('col-pad')
    # driver.quit()
    return outputElement.text


def record_connection_time(url):
    millis = int(round(time.time() * 1000))
    driver = webdriver.Chrome(DRIVER)
    driver.get(url)
    millis2 = int(round(time.time() * 1000))
    return millis2 - millis


# one problem here is that, after the first query, it begins to
# be cached and does not really show the real result
def record_DNS_TTL(url):
    hostname = urlparse(url).hostname
    answer = dns.resolver.query(hostname)
    return answer.rrset.ttl


# print(crawlRanking(URL))
print record_DNS_TTL(URL)
