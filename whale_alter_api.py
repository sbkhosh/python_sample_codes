#!/usr/bin/python3

import os
import requests

from pprint import pprint

api_key = os.getenv('WHALE_API_KEY')
response = requests.get('https://api.whale-alert.io/v1/status?api_key={}'.format(api_key))
resp = response.json()
pprint(resp)

