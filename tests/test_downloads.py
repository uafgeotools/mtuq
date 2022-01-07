#!/usr/bin/env python


from obspy import UTCDateTime
from mtuq.io.clients.fdsn import download_waveforms
from mtuq.util import fullpath


if __name__=='__main__':

    origin_id = '20090407201255351'

    path_waveforms = fullpath('data/waveforms/FDSN/', origin_id, 'waveforms')
    path_stationxml= fullpath('data/waveforms/FDSN/', origin_id, 'stationxml')


    download_waveforms(
        origin_time = UTCDateTime('2009-04-07T20:12:55.000000Z'),
        origin_latitude = 61.454200744628906,
        origin_longitude = -149.7427978515625,
        origin_depth_in_m = 33033.599853515625,
        starttime_in_s = -100,
        endtime_in_s = 300,
        mseed_storage = path_waveforms,
        stationxml_storage = path_stationxml,
        radius_in_km = 1000.,
        )



