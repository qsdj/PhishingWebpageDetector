
""" Requirement:
    (1) Internet connection
    (2) Levenshtein package for computing the distance between 2 strings
    (2) whois package for domain-based feature garthering
    (3) validators packages for URL validation
    (4) dnspython packages for DNS queries
    (5) pygeoip packages for geolocation information
    (6) GeoIP.dat file


"""

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
import dns.resolver
import sys


sample_url = "https://www.ultimatesoftware.com/"
#sample_url = "https://enphg.ultiprotime.com/reports/cognos/submitParams.jsp?report_name=DAILY+LABOR+PERCENT+OF+SALES+REPORT&parameterTarget=cognosServer"
#sample_url = "http://shoppingdaimpressora.com.br/downloader/skin/ew34.ultipro.com.html?q=abc&p=123"

product_domain_name = 'ultipro.com'

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
suspicious_TLDs = ['.cc', '.pw', '.tk', '.info', '.net', '.ga', '.top', '.ml', '.cf', '.cn', '.gq', '.ve', '.kr', '.it', '.hu', '.es']
generic_TLDs = ['.gov', '.edu', '.org']
company_IPs = []


def getEntropy(f_array):
    """ To calculate the entropy (the level of randomness) of a string. """

    C = collections.Counter(f_array)
    counts  = np.array(C.values(), dtype=float)
    probs = counts/counts.sum()
    entropy = ( -probs * np.log2(probs) ).sum()
    return entropy  # a float


def KLDistance():
    """ To calculate the Kullback-Leibler Divergence between 2 strings. """



    return -1


def KSDistance():
    """ To calculate the Kolmogorove-Smirmov Distance between 2 strings. """

    return -1



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
        return [float(sum_len)/word_count, word_count, largest_size, words]
    except:
        return [0, word_count, largest_size, words]


def containIP(word_list):
    """ To check that there is an IP address in the input list or not. """

    count = 0;
    for word in word_list:
        if unicode(word).isnumeric():
            count += 1
        else:
            if count >= 4 :
                return 1
            else:
                count = 0;
    if count >= 4:
        return 1

    return 0


def find_element_with_attribute(dom, element, attribute):
    for subelement in dom.getElementsByTagName(element):
        if subelement.hasAttribute(attribute):
            return subelement.attributes[attribute].value
    return -1


def getPagePopularity(host):
    """ To get the Alexa ranks of the input domain, namely the global rank and the country rank respectively. """

    xmlpath = 'http://data.alexa.com/data?cli=10&dat=s&url='+host

    try:
        xml = urllib2.urlopen(xmlpath)
        dom = minidom.parse(xml)
        global_rank = find_element_with_attribute(dom, 'REACH', 'RANK')
        country_rank = find_element_with_attribute(dom, 'COUNTRY', 'RANK')
        return [global_rank, country_rank]
    except:
        return [-1, -1]


def FeatureExtraction(url_input):
    """ To extract the following features from a URL:
        (1) lexical Features,
        (2) page-based Features,
        (3) host/domain-based Features.

    """



    if validators.url(url_input):
        is_url = True
        print "Extracting features of URL: ", url_input
    else:
        is_url = False
        print "Extracting features of domain: ", url_input

    Features = {}

    print "  Extracting common lexical features..."
    if(is_url):
        # lexical Features - length
        Features['record_type'] = "url"
        Features['url'] = url_input
        Features['url_length'] = len(url_input)

        obj = urlparse(url_input)
        Features['protocol'] = obj.scheme
        Features['is_http'] = 1 if obj.scheme == "http" else 0
        Features['is_https'] = 1 if obj.scheme == "https" else 0

        path = obj.path
        Features['url_path'] = obj.path
        Features['url_path_length'] = len(obj.path)
        Features['url_path_depth'] = obj.path.count('/')

        Features['url_params'] = obj.params
        Features['url_params_length'] = len(obj.params)

        # lexical features - argument features
        query = obj.query
        Features['url_query'] = obj.query
        Features['url_query_argument_count'] = len(query.split('&'))
        Features['url_query_length'] = len(obj.query)

        Features['url_fragment'] = obj.fragment

        fqdn = obj.netloc
        Features['fqdn'] = fqdn   # network location (i.e. www.hostname.domain.com)
        Features['fqdn_length'] = len(fqdn)

        # lexical features - subdomain count
        subdomains = fqdn.split('.')
        TLD = subdomains[-1]
        Features['url_TLD'] = TLD
        domain = ".".join(subdomains[-2:])

        Features['url_path_avg_token_length'], Features['url_path_token_count'], Features['url_path_largest_token_size'], path_tokens = tokenise(path)
        #Features['url_path_levenshtein_distance'] = Levenshtein.distance(path, product_domain_name)

        # lexical features - the ratio of path length and hostname length
        Features['url_path_len_and_hostname_len_ratio'] = float(len(path)) / float(len(fqdn)) if len(fqdn) > 0 else -1

        for n in product_domain_names:
            name = ''
            name = 'url_path_levenshtein_distance_' + n
            Features[name] = Levenshtein.distance(path, n)


    else:   # a domain name
        Features['record_type'] = "domain"
        Features['url'] = url_input
        Features['url_length'] = len(url_input)

        Features['protocol'] = -1
        Features['is_http'] = -1
        Features['is_https'] = -1

        Features['url_path'] = -1
        Features['url_path_length'] = -1
        Features['url_path_depth'] = -1

        Features['url_params'] = -1
        Features['url_params_length'] = -1

        Features['url_query'] = -1
        Features['url_query_argument_count'] = -1
        Features['url_query_length'] = -1

        Features['url_fragment'] = -1

        fqdn = url_input
        Features['fqdn'] = url_input    # network location (i.e. www.hostname.domain.com)
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

        Features['url_path_levenshtein_distance'] = -1

        # lexical features - the ratio of path length and hostname length
        Features['url_path_len_and_hostname_len_ratio'] = -1



    Features['url_domain'] = domain
    subdomain_count = len(subdomains) - 1   # exclude the TLD
    Features['is_dotcom'] = 1 if TLD == 'com' else 0
    Features['subdomain_count'] = subdomain_count

    # lexical features - Bag-Of-Word (BOW) counts
    Features['url_dot_count'] = url_input.count('.')
    Features['url_dash_count'] = url_input.count('-')
    Features['url_plus_count'] = url_input.count('+')
    Features['url_at_count'] = url_input.count('@')
    Features['url_http_count'] = url_input.count('http')
    Features['url_www_count'] = url_input.count('www')
    Features['url_digit_count'] = sum( c.isdigit() for c in url_input )
    Features['url_alphabet_count'] = sum (c.isalpha() for c in url_input )
    Features['url_non_alphabet_count'] = len(url_input) - Features['url_alphabet_count'] - Features['url_digit_count']



    # lexical features - TLD counts
    for tld in company_TLDs:
        fname = ''
        fname = 'url_' + tld[1:] + '_count'
        Features[fname] = url_input.count(tld)

    for tld in suspicious_TLDs:
        fname = ''
        fname = 'url_' + tld[1:] + '_count'
        Features[fname] = url_input.count(tld)

    for tld in generic_TLDs:
        fname = ''
        fname = 'url_' + tld[1:] + '_count'
        Features[fname] = url_input.count(tld)

    # lexical features - brandname / product name existence
    for n in brand_names:
        name = ''
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    for n in product_names:
        name = ''
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    for n in suspicious_brand_names:
        name = ''
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    for n in suspicious_product_names:
        name = ''
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    for n in suspicious_words:
        name = ''
        name = 'url_contain_' + n
        Features[name] = 1 if n in url_input else 0

    # lexical features - token counts
    Features['url_avg_token_length'], Features['url_token_count'], Features['url_largest_token_size'], url_tokens = tokenise(url_input)
    Features['fqdn_avg_token_length'], Features['fqdn_token_count'], Features['fqdn_largest_token_size'], fqdn_tokens = tokenise(fqdn)

    # is it IP?
    Features['is_IP'] = containIP(url_tokens)

    # lexical features - the randomness
    Features['url_entropy'] = getEntropy(list(url_input))

    # lexical features - the similarity
    Features['url_levenshtein_distance'] = Levenshtein.distance(url_input, product_domain_name)
    Features['fqdn_levenshtein_distance'] = Levenshtein.distance(fqdn, product_domain_name)



    # lexical features - typosquatting measure

    # host/domain-based features
    # host/domain-based features - WHOIS Features

    print "  Extracting WHOIS features thru the Internet..."
    try:
        whois_res = whois.query(domain)
        Features['domain_name'] = whois_res.name
        Features['domain_expiration_date'] = whois_res.expiration_date
        Features['domain_last_updated'] = whois_res.last_updated
        Features['domain_registrar'] = whois_res.registrar
        Features['domain_creation_date'] = whois_res.creation_date
        cur_datetime = datetime.datetime.now()
        Features['domain_current_date'] = cur_datetime

        # how many days passed since the domain was
        if whois_res.creation_date != None:
            domain_active_time = cur_datetime - whois_res.creation_date
        else:
            domain_active_time = -1
            domain_active_time.days = -1
        Features['domain_active_time'] = domain_active_time
        Features['domain_active_days'] = domain_active_time.days


        if whois_res.expiration_date is not None:
            TTL = whois_res.expiration_date - cur_datetime
            Features['domain_TTL'] = TTL
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
        Features['domain_current_date'] = cur_datetime

        # how many days passed since the domain was
        domain_active_time = -1
        Features['domain_active_time'] = -1
        Features['domain_active_days'] = -1

        Features['domain_TTL'] = -1
        Features['domain_TTL_in_days'] = -1


    # host/domain-based features - DNS answers
    print "  Extracting DNS answer features thru the Internet..."
    IPs = []
    try:
        dns_resolver = dns.resolver.Resolver()
        dns_answer = dns_resolver.query(domain, "A")    # look up for "A" record(s) for the domain

        for rdata in dns_answer:
            IPs.append(rdata.address)

        Features['IPs'] = IPs
        Features['IP_count'] = len(IPs)

    except:
        Features['IPs'] = []
        Features['IP_count'] = 0

    # host/domain-based features - geolocation features
    print "  Extracting geolocation features thru the Internet..."

    try:
        gi = pygeoip.GeoIP('GeoIP.dat')
        Features['domain_country_code_by_name'] = gi.country_code_by_name(domain)
        Features['domain_country_name_by_name'] = gi.country_name_by_name(domain)
    except:
        Features['domain_country_code_by_name'] = -1
        Features['domain_country_name_by_name'] = -1

    # host/domain-based features - ASN information
    try:
        gi = pygeoip.GeoIP('GeoIPASNum.dat')
        Features['domain_asn_by_name'] = gi.asn_by_name(domain)
        Features['domain_asn_no_by_name'] = int(gi.asn_by_name(domain).split()[0][2:])
    except:
        #Features['domain_asn'] = -1
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


    # page-based Features
    print "  Extracting page-based features thru the Internet..."
    #Features['page_global_rand'], Features['page_country_rank'] = getPagePopularity("google.com")
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


print FeatureExtraction(sample_url)



def main():

    if len(sys.argv) < 3:
        print "Usage: python urls2features.py domains.csv"
        sys.exit()

    list_of_feature_sets = []
    #list_of_dicts = [{'a': 1, 'b': 2}, {'b': 4, 'a': 3}]
    count = 1
    url_features = FeatureExtraction('ultimatesoftware.com')
    feature_names = url_features.keys()

    with open(sys.argv[2], 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=feature_names)
        writer.writeheader()


    #url_features = FeatureExtraction(sample_url)

    #for line in open("alexa_top_1k_domains.csv"):
    for line in open(sys.argv[1]):
        line = line.rstrip('\n')
        url_features = FeatureExtraction(line)

        with open(sys.argv[2], 'a') as csv_file:
        #with open("alexa_top_1k_domain_features.csv", 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=feature_names)
            try:
                writer.writerow(url_features)
                csv_file.flush()
            except:
                print "Error writing to the CSV file"
            #for data in list_of_feature_sets:
                #writer.writerow(data)

        #list_of_feature_sets.append(url_features)
        print "#total_domains extracted: ", count
        count = count+1

    # save to a CSV file



main()
