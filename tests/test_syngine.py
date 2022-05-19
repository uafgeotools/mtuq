#!/usr/bin/env python3

from urllib.request import URLopener

syngine_url = 'http://service.iris.edu/irisws/syngine/1/query?model=ak135f_2s&dt=0.02&greensfunction=1&sourcedistanceindegrees=0.1425432904952431&sourcedepthinmeters=25000&origintime=2009-04-07T20:12:55.000000&starttime=2009-04-07T20:12:55.000000'

opener = URLopener()
opener.retrieve(syngine_url, 'output.zip')

