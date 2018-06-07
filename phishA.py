#!/usr/bin/python
"""
    The aim of this script is to discover possible phishing sites that exist in isslogs in 30 days ago.

    This script are to:
    (1) mainly query the hostname and the referer of a web transation from iislogs using a Tidal query named iis-phishing-search.
        Note: this script will query data from the data centre (such as ALT or PHX) that you set in your environment
    (2) compare the hostname and referer of the transaction, if the domain of the referer is not the hostname, the referer domain may be a phishing site. The transation is malicious.
        Otherwise, the transaction is legitimate.

"""

import re
import json
import os
import sys
from datetime import datetime
from tidal.client import TidalClient
import pandas as pd
from urlparse import urlparse
import csv

tidal = TidalClient()

whitelist = []
searchEngines = []

#https://en.wikipedia.org/wiki/List_of_search_engines

def main():
    # default values
    num_records = 10000   # specify the maximum number of queried records to not overwhelm Elasticsearch
    num_days = 30

    todate = sys.argv[1]
    fromdate = sys.argv[2]
    # query data
    tidal = TidalClient()

    res = tidal.query("iis-search-phishing", {"size": num_records, "todate": todate, "fromdate": fromdate})

    #print json.dumps(res)

    unmatched = []
    matched_counter = 0

    for rec in res["hits"]["hits"]:
        pair = {}

        # get only the domain name
        parsed_uri = urlparse(rec['_source']['referer'])
        ref_domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

        if "http://" in ref_domain:
            ref_domain = ref_domain.replace("http://", "")

        if "https://" in ref_domain:
            ref_domain = ref_domain.replace("https://", "")

        # cut the last character "/"
        ref_domain = ref_domain[:-1]

        if rec['_source']['hostname'] not in ref_domain:
            pair['hostname'] = rec['_source']['hostname']
            pair['referer'] = rec['_source']['referer']
            pair['timestamp'] = rec['_source']['@timestamp']
            pair['ref_domain'] = ref_domain
            pair['status'] = rec['_source']['status']
            #pair['product'] = rec['_source']['product']
            pair['method'] = rec['_source']['method']
            pair['request'] = rec['_source']['request']
            pair['clientip'] = rec['_source']['clientip']
            pair['computername'] = rec['_source']['computername']
            unmatched.append(pair)
        else:
            matched_counter = matched_counter + 1

    print "#matched_records: ", matched_counter
    print "#unmatched_records: ", len(unmatched)
    #print {unmatched}
    keys = unmatched[0].keys()
    filename = fromdate + ".csv"
    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(unmatched)

if __name__ == "__main__":
    main()
    #if len(sys.argv) < 2:
        #sys.exit()
    #else:
        #main()
