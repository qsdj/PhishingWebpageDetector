""" Requirement:
    (1) Internet connection
    (2) Levenshtein package for computing the distance between 2 strings
    (2) whois package for domain-based feature garthering
    (3) validators packages for URL validation
    (4) dnspython packages for DNS queries
    (5) pygeoip packages for geolocation information
    (6) GeoIP.dat, GeoIPASNum.dat, GeoLiteCity.dat files


"""
import itertools
from urlparse import urlparse
import re
import urllib2
import urllib
from xml.dom import minidom
import csv
import pygeoip
import collections
import numpy as np
import Levenshtein
import whois
import validators
import datetime
from datetime import date
import dns.resolver
import sys
import json
import sklearn
import time
import pyxdameraulevenshtein as dl
import pyjarowinkler.distance as jw
from nltk import ngrams
import scipy
import wordninja

# Usage of Damerau Levenshtein: use dl.damerau_levenshtein_distance or dl.normalized_damerau_levenshtein_distance
# Usage of Jaro Winkler: use pw.get_jaro_distance


DRIVER = '/Users/usgxchen/Downloads/chromedriver'
sample_url = "https://www.ultimatesoftware.com/"
# sample_url = "https://enphg.ultiprotime.com/reports/cognos/submitParams.jsp?report_name=DAILY+LABOR+
# PERCENT+OF+SALES+REPORT&parameterTarget=cognosServer"
# sample_url = "http://shoppingdaimpressora.com.br/downloader/skin/ew34.ultipro.com.html?q=abc&p=123"

product_domain_names = ['ultipro.com']

# defined our brandnames and product names
brand_names = ["ultimatesoftware"]
suspicious_brand_names = ["ulti", "software", "soft", "sw"]
suspicious_product_names = ["pro", "upro"]
suspicious_words = ["login", "signin", "confirm", "account", "secure", "banking"]
product_names = ["ultipro", "ultihome"]
blacklist_IPs = []
whitelist_IPs = []
blacklist_domainnames = []
whitelist_domainnames = []
blacklist_hostnames = []
whitelist_hostnames = []
company_TLDs = ['.com']
suspicious_TLDs = ['.cc', '.pw', '.tk', '.info', '.net', '.ga', '.top', '.ml', '.cf', '.cn', '.gq', '.ve', '.kr', '.it',
                   '.hu', '.es']
generic_TLDs = ['.gov', '.edu', '.org']
company_IPs = []


def json_serial(obj):
    """ JSON serializer for objects not serializable by default json code """

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def getEntropy(f_array):
    """
    To calculate the entropy (the level of randomness) of a string.
    :param f_array: the array whose entropy is to be calculated
    :return: the entropy as a float
    """

    C = collections.Counter(f_array)
    counts = np.array(C.values(), dtype=float)
    probs = counts / counts.sum()
    entropy = (-probs * np.log2(probs)).sum()
    return entropy  # a float


def KLDistance():
    """ To calculate the Kullback-Leibler Divergence between 2 strings. """

    return -1


def KSDistance():
    """ To calculate the Kolmogorove-Smirmov Distance between 2 strings. """

    return -1


def lcs_length(a, b):
    """
    Calculate the longest common sequence in two strings
    :param a: 1st string
    :param b: 2nd string
    :return: an integer indicating the longest common sequence
    """

    table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def ngram_damerau_levenshtein(str1, str2, n):
    """
    n-gram damerau levenshtein distance, cuts string into length n adjacent lists
    and treat them as characters
    :param str1: 1st string
    :param str2: 2nd string
    :param n: size of adjacent list
    :return: the n-gram damerau levenshtein distance as a float
    """
    # still need to keep reading the paper at
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.9369&rep=rep1&type=pdf
    ngram1 = ngrams(str1, n)
    ngram2 = ngrams(str2, n)
    return lcs_length(list(ngram1), list(ngram2))


def kolmogorov_complexity(str1, str2):
    # This is an exponential problem: https://codegolf.stackexchange.com/questions/47353/write-a-kolmogorov
    # -complexity-solver
    """not functional"""
    return scipy.stats.kstest(str1, str2)


def tokenise(input_str):
    """ To tokenise the input string into words (tokens).

    Returns: 4 values, namely
        the avarage length of words,
        the number of words,
        the largest size of word,
        the list of words (tokens) respectively.

    """

    if input_str == '':
        return [0, 0, 0, []]

    words = re.split('\W+', input_str)
    word_count = sum_len = largest_size = 0

    for word in words:
        l = len(word)
        sum_len += l
        if l > 0:
            word_count += 1
        if largest_size < l:
            largest_size = l

    try:
        return [float(sum_len) / word_count, word_count, largest_size, words]
    except:
        return [0, word_count, largest_size, words]


def containIP(word_list):
    """
    To check that there is an IP address in the input list or not.
    :param: word_list: the list of words to be checed
    :returns: 1 if it contains an IP, 0 otherwise

    """

    count = 0
    for word in word_list:
        if unicode(word).isnumeric():
            count += 1
        else:
            if count >= 4:
                return 1
            else:
                count = 0
    if count >= 4:
        return 1

    return 0


def find_element_with_attribute(dom, element, attribute):
    for subelement in dom.getElementsByTagName(element):
        if subelement.hasAttribute(attribute):
            return subelement.attributes[attribute].value
    return -1


def get_path_host_ratio(url):
    """
    :param url
    :return: # of characters in path/ # of characters in hostname
    """
    path = urlparse(url).path
    hostname = urlparse(url).hostname
    return len(path) / len(hostname)


def getPagePopularity(host):
    """
     To get the Alexa ranks of the input domain, namely the global rank and the country rank respectively.
    :param host: the hostname to be searched on alexa
    :return: [global rankings, country rankings], [-1, -1] if not found
    """

    xmlpath = 'http://data.alexa.com/data?cli=10&dat=s&url=' + host

    try:
        xml = urllib2.urlopen(xmlpath)
        dom = minidom.parse(xml)
        global_rank = find_element_with_attribute(dom, 'REACH', 'RANK')
        country_rank = find_element_with_attribute(dom, 'COUNTRY', 'RANK')
        return [global_rank, country_rank]
    except:
        return [-1, -1]


def url_feature_extraction(url_input):
    """
    features extracted:
        features:
            protocol: the network protocol run by the page
            is_http: 1 if starts with http, 0 otherwise
            is_https: 1 if starts with https, 0 otherwise
            url_path: path
            url_path_length: # of chars
            url_path_depth: # of '/'
            url_params:
            url_params_length:
            url_query:
            url_query_argument_count: # of arguments in the query
            url_query_length: # of chars
            ...
    :param url_input: url from which we extract features
    :return: Features[], domain, subdomains[]
    """

    Features = {'record_type': "url", 'url': url_input, 'url_length': len(url_input)}

    # lexical Features - length

    url_obj = urlparse(url_input)
    Features['protocol'] = url_obj.scheme
    Features['is_http'] = 1 if url_obj.scheme == "http" else 0
    Features['is_https'] = 1 if url_obj.scheme == "https" else 0

    path = url_obj.path
    Features['url_path'] = url_obj.path
    Features['url_path_length'] = len(url_obj.path)
    Features['url_path_depth'] = url_obj.path.count('/')

    Features['url_params'] = url_obj.params
    Features['url_params_length'] = len(url_obj.params)

    # lexical features - argument features
    query = url_obj.query
    Features['url_query'] = url_obj.query
    Features['url_query_argument_count'] = len(query.split('&'))
    Features['url_query_length'] = len(url_obj.query)

    Features['url_fragment'] = url_obj.fragment

    fqdn = url_obj.netloc
    Features['fqdn'] = fqdn  # network location (i.e. www.hostname.domain.com)
    Features['fqdn_length'] = len(fqdn)

    # lexical features - subdomain count
    subdomains = fqdn.split('.')
    TLD = subdomains[-1]
    Features['url_TLD'] = TLD
    domain = ".".join(subdomains[-2:])

    Features['url_path_avg_token_length'], Features['url_path_token_count'], Features[
        'url_path_largest_token_size'], path_tokens = tokenise(path)

    for n in product_domain_names:
        name = 'url_path_levenshtein_distance_' + n
        Features[name] = Levenshtein.distance(path, n)

    # Features['url_path_levenshtein_distance'] = Levenshtein.distance(path, product_domain_name)

    # lexical features - the ratio of path length and hostname length
    Features['url_path_len_and_hostname_len_ratio'] = float(len(path)) / float(len(fqdn)) if len(fqdn) > 0 else -1

    return Features, domain, subdomains


def domain_feature_extraction(url_input):
    """
    Extracts features from domain
    :param url_input
    :return: Features[], domain, subdomains[]
    """
    Features = {'record_type': "domain", 'url': url_input, 'url_length': len(url_input), 'protocol': -1, 'is_http': -1,
                'is_https': -1, 'url_path': -1, 'url_path_length': -1, 'url_path_depth': -1, 'url_params': -1,
                'url_params_length': -1, 'url_query': -1, 'url_query_argument_count': -1, 'url_query_length': -1,
                'url_fragment': -1}

    fqdn = url_input
    Features['fqdn'] = url_input  # network location (i.e. www.hostname.domain.com)
    Features['fqdn_length'] = len(url_input)
    domain = url_input

    # lexical features - subdomain count
    subdomains = fqdn.split('.')
    TLD = subdomains[-1]
    Features['url_TLD'] = TLD
    Features['url_path_avg_token_length'] = -1
    Features['url_path_token_count'] = -1
    Features['url_path_largest_token_size'] = -1
    path_tokens = []

    # Features['url_path_levenshtein_distance'] = -1

    for n in product_domain_names:
        name = 'url_path_levenshtein_distance_' + n
        Features[name] = -1

    # lexical features - the ratio of path length and hostname length
    Features['url_path_len_and_hostname_len_ratio'] = -1
    return Features, domain, subdomains


def tld_search(Features, url_input):
    """
    Go through all TLDs, store the number of appearances in a dictionary, add it to Features
    :param Features: input dictionary to operate upon
    :param url_input
    :return: Features: dictionary after adding the TLD counts
    """
    for tld in itertools.chain(company_TLDs, suspicious_TLDs, generic_TLDs):
        fname = 'url_' + tld[1:] + '_count'
        Features[fname] = url_input.count(tld)

    return Features


def name_search(Features, url_input):
    """
        Go through all names, store the number of appearances in a dictionary, add it to Features
        :param Features: input dictionary to operate upon
        :param url_input
        :return: Features: dictionary after adding the name counts
        """
    for n in itertools.chain(brand_names, product_names, suspicious_brand_names, suspicious_product_names,
                             suspicious_words):
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    return Features


def hamming(str1, str2):
    """
    Calculate Hamming distance
    :precondition: strings are of the same size
    :param str1: 1st string
    :param str2: 2nd string
    :return: hamming distance
    """
    if len(str1) == len(str2):
        return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))
    else:
        raise Exception("Trying to calculate hamming distance when strings are not of same length")


def get_similarities(Features, url_input):
    """
    similarity metrics include: Levenshtein, jaro, damerau levenshtein, normalized_damerau_levenshtein,
    and hamming distance
    :param Features: input dictionary to add things to
    :param url_input
    :return: Features: after adding all similarity metrics
    """
    for n in itertools.chain(product_domain_names, brand_names):
        Features['url_levenshtein_distance_' + n] = Levenshtein.distance(url_input, n)
        Features['fqdn_levenshtein_distance_' + n] = Levenshtein.distance(Features['fqdn'], n)
        Features['url_jaro_winkler_distance_' + n] = jw.get_jaro_distance(url_input, n)
        Features['fqdn_jaro_winkler_distance_' + n] = jw.get_jaro_distance(Features['fqdn'], n)
        Features['url_damerau_levenshtein_distance_' + n] = dl.damerau_levenshtein_distance(url_input, n)
        Features['fqdn_damerau_levenshtein_distance_' + n] = dl.damerau_levenshtein_distance(Features['fqdn'], n)
        Features['url_damerau_levenshtein_normalized_distance_' + n] = dl.normalized_damerau_levenshtein_distance(
            url_input, n)
        Features['fqdn_damerau_levenshtein_normalized_distance_' + n] = dl.normalized_damerau_levenshtein_distance(
            Features['fqdn'], n)
        if len(n) == len(url_input):
            Features['url_length_equals_' + n] = 1
            Features['url_hamming_distance_' + n] = hamming(url_input, n)
            Features['fqdn_hamming_distance_' + n] = hamming(Features['fqdn'], n)
        else:
            Features['url_length_equals_' + n] = 0
    return Features


# noinspection PyBroadException
def get_whois_features(Features, domain):
    """
    DNS/IP based features mostly
    :param Features:
    :param domain:
    :return:
    """
    try:
        whois_res = whois.query(domain)
        Features['domain_name'] = whois_res.name
        Features['domain_expiration_date'] = whois_res.expiration_date
        Features['domain_last_updated'] = whois_res.last_updated
        Features['domain_registrar'] = whois_res.registrar
        Features['domain_creation_date'] = whois_res.creation_date
        cur_datetime = str(datetime.datetime.now())
        Features['domain_current_date'] = str(cur_datetime)

        # how many days passed since the domain was
        if whois_res.creation_date is not None:
            domain_active_time = cur_datetime - whois_res.creation_date
        else:
            domain_active_time = -1
            domain_active_time.days = -1
        Features['domain_active_time'] = str(domain_active_time)
        Features['domain_active_days'] = domain_active_time.days

        if whois_res.expiration_date is not None:
            TTL = whois_res.expiration_date - cur_datetime
            Features['domain_TTL'] = str(TTL)
            Features['domain_TTL_in_days'] = TTL.days
        else:
            Features['domain_TTL'] = -1
            Features['domain_TTL_in_days'] = -1
    except:
        Features['domain_name'] = domain
        Features['domain_expiration_date'] = -1
        Features['domain_last_updated'] = -1
        Features['domain_registrar'] = -1
        Features['domain_creation_date'] = -1
        cur_datetime = datetime.datetime.now()
        Features['domain_current_date'] = str(cur_datetime)

        # how many days passed since the domain was
        Features['domain_active_time'] = -1
        Features['domain_active_days'] = -1

        Features['domain_TTL'] = -1
        Features['domain_TTL_in_days'] = -1

    return Features


# noinspection PyBroadException
def get_dns_answer_features(Features, domain):
    """
    DNS answer features
    :param Features:
    :param domain:
    :return:
    """
    IPs = []
    try:
        dns_resolver = dns.resolver.Resolver()
        dns_answer = dns_resolver.query(domain, "A")  # look up for "A" record(s) for the domain

        for rdata in dns_answer:
            IPs.append(rdata.address)

        Features['IPs'] = IPs
        Features['IP_count'] = len(IPs)

    except:
        Features['IPs'] = []
        Features['IP_count'] = 0

    return Features


# noinspection PyBroadException
def get_asn_information(Features, domain):
    try:
        gi = pygeoip.GeoIP('GeoIPASNum.dat')
        Features['domain_asn_by_name'] = gi.asn_by_name(domain)
        Features['domain_asn_no_by_name'] = int(gi.asn_by_name(domain).split()[0][2:])
    except:
        # Features['domain_asn'] = -1
        Features['domain_asn_by_name'] = -1
        Features['domain_asn_no_by_name'] = -1

    try:
        gi = pygeoip.GeoIP('GeoLiteCity.dat')
        gi_record = gi.record_by_name(domain)
        Features['domain_city_by_name'] = gi_record['city']
        Features['domain_city_country_code_by_name'] = gi_record['country_code']
        Features['domain_area_code_by_name'] = gi_record['area_code']
        Features['domain_continent_by_name'] = gi_record['continent']
        Features['domain_country_code_by_name_2'] = gi_record['country_code']
        Features['domain_region_code_by_name'] = gi_record['region_code']
        Features['domain_time_zone_by_name'] = gi_record['time_zone']
        Features['domain_country_name_by_name_2'] = gi_record['country_name']
    except:
        Features['domain_city_by_name'] = -1
        Features['domain_city_country_code_by_name'] = -1
        Features['domain_area_code_by_name'] = -1
        Features['domain_continent_by_name'] = -1
        Features['domain_country_code_by_name_2'] = -1
        Features['domain_region_code_by_name'] = -1
        Features['domain_time_zone_by_name'] = -1
        Features['domain_country_name_by_name_2'] = -1

    return Features


def get_geoip(Features, domain):
    try:
        gi = pygeoip.GeoIP('GeoIP.dat')
        Features['domain_country_code_by_name'] = gi.country_code_by_name(domain)
        Features['domain_country_name_by_name'] = gi.country_name_by_name(domain)
    except:
        Features['domain_country_code_by_name'] = -1
        Features['domain_country_name_by_name'] = -1
    return Features


def BOW_distances(Features, other_url):
    # picking the features excluding www and other generic elements that URLs have
    a = [Features['url_dot_count'], Features['url_dash_count'], Features['url_plus_count'], Features['url_at_count'],
         Features['url_http_count'], Features['url_www_count'], Features['url_digit_count'],
         Features['url_alphabet_count'], Features['url_non_alphabet_count']]
    other = {}
    other = get_BOW_info(other, other_url)
    b = [other['url_dot_count'], other['url_dash_count'], other['url_plus_count'], other['url_at_count'],
         other['url_http_count'], other['url_www_count'], other['url_digit_count'],
         other['url_alphabet_count'], other['url_non_alphabet_count']]
    avec = np.ndarray(a)
    bvec = np.ndarray(b)
    fakeJaccard = sklearn.metrics.jaccard_similarity_score(avec, bvec)
    euclidean = np.linalg.norm(avec - bvec)
    Features['euclidean_' + other[other_url]] = euclidean
    Features['fakeJaccard_' + other[other_url]] = fakeJaccard
    return Features


def get_BOW_info(Features, url_input):
    """
    retrieve bag of words information from the url
    :param Features: dictionary where we store the information
    :param url_input:
    :return: Features: after adding the information
    """
    Features['url_dot_count'] = url_input.count('.')
    Features['url_dash_count'] = url_input.count('-')
    Features['url_plus_count'] = url_input.count('+')
    Features['url_at_count'] = url_input.count('@')
    Features['url_http_count'] = url_input.count('http')
    Features['url_www_count'] = url_input.count('www')
    Features['url_digit_count'] = sum(c.isdigit() for c in url_input)
    Features['url_alphabet_count'] = sum(c.isalpha() for c in url_input)
    Features['url_non_alphabet_count'] = len(url_input) - Features['url_alphabet_count'] - Features['url_digit_count']

    return Features


def get_typosquatting_measure(Features, url_input):
    """not functional"""
    return Features


def get_lexical_features(Features, url_input):
    """
    Get all lexical features through calling above defined functions, individual lines are explained
    :param Features:
    :param url_input:
    :return: Features
    """

    # lexical features - Bag-Of-Word (BOW) counts
    Features = get_BOW_info(Features, url_input)

    # lexical features - TLD counts
    Features = tld_search(Features, url_input)

    # lexical features - brandname / product name existence
    Features = name_search(Features, url_input)

    # lexical features - token counts
    Features['url_avg_token_length'], Features['url_token_count'], Features[
        'url_largest_token_size'], url_tokens = tokenise(url_input)
    Features['fqdn_avg_token_length'], Features['fqdn_token_count'], Features[
        'fqdn_largest_token_size'], fqdn_tokens = tokenise(Features['fqdn'])

    # is it IP?
    Features['is_IP'] = containIP(url_tokens)

    # lexical features - the randomness
    Features['url_entropy'] = getEntropy(list(url_input))

    # lexical features - the similarity
    Features = get_similarities(Features, url_input)

    # lexical features - typosquatting measure
    Features = get_typosquatting_measure(Features, url_input)
    return Features


def get_host_domain_features(Features, domain):
    """
        Get all domain features through calling above defined functions, individual lines are explained
        :param Features:
        :param domain:
        :return: Features
    """
    # host/domain-based features - WHOIS Features
    # print "  Extracting WHOIS features thru the Internet..."
    Features = get_whois_features(Features, domain)

    # host/domain-based features - DNS answers
    # print "  Extracting DNS answer features thru the Internet..."
    Features = get_dns_answer_features(Features, domain)

    # host/domain-based features - geolocation features
    # print "  Extracting geolocation features thru the Internet..."
    Features = get_geoip(Features, domain)

    # host/domain-based features - ASN information
    Features = get_asn_information(Features, domain)

    return Features


def FeatureExtraction(url_input):
    """ To extract the following features from a URL:
        (1) lexical Features,
        (2) page-based Features,
        (3) host/domain-based Features.

    """
    if validators.url(url_input):
        is_url = True
        # print "Extracting features of URL: ", url_input
    else:
        is_url = False
        # print "Extracting features of domain: ", url_input

    # print "  Extracting common lexical features..."
    if is_url:
        Features, domain, subdomains = url_feature_extraction(url_input)

    else:  # a domain name
        Features, domain, subdomains = domain_feature_extraction(url_input)

    Features['url_domain'] = domain
    subdomain_count = len(subdomains) - 1  # exclude the TLD
    Features['is_dotcom'] = 1 if Features['url_TLD'] == 'com' else 0
    Features['subdomain_count'] = subdomain_count

    # lexical features
    Features = get_lexical_features(Features, url_input)

    # host/domain-based features
    Features = get_host_domain_features(Features, domain)

    # page-based Features
    # print "  Extracting page-based features thru the Internet..."
    # Features['page_global_rand'], Features['page_country_rank'] = getPagePopularity("google.com")
    alexa_global_rank, alexa_country_rank = getPagePopularity(domain)
    Features['page_global_rank'] = alexa_global_rank
    Features['page_country_rank'] = alexa_country_rank
    if alexa_global_rank == -1:
        Features['page_is_Alexa_top1m'] = -1
    else:
        Features['page_is_Alexa_top1m'] = 1 if alexa_global_rank > 10000000 else 0

    # content-based features

    """
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(Features)

    print "#features = ", len(Features.keys())
    """

    return Features


def main():
    """
    dump the features into a json file
    :return: None
    """
    if len(sys.argv) < 2:
        url_features = FeatureExtraction(sample_url)
    else:
        url_features = FeatureExtraction(sys.argv[1])

    print json.dumps(url_features)

# main()
