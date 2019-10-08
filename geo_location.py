#!/usr/bin/python3

from geopy.geocoders import Nominatim
import math
import numpy as np
import geocoder

def get_dist_lat_lon_km(lat1,lon1,lat2,lon2):
  R = 6371
  dLat = deg2rad(lat2-lat1)
  dLon = deg2rad(lon2-lon1); 
  a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) *  math.sin(dLon/2) * math.sin(dLon/2)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  d = R * c
  return(d)

def deg2rad(deg):
  return(deg * np.pi/180)

def get_distance(adr1, adr2):
    geolocator = Nominatim(user_agent="")
    loc1 = geolocator.geocode(adr1)
    loc2 = geolocator.geocode(adr2)
    res = get_dist_lat_lon_km(loc1.latitude, loc1.longitude, loc2.latitude, loc2.longitude)
    print("Distance between {adrs1} and {adrs2} is {distance}".format(adrs1=adr1, adrs2=adr2, distance=res))

          
adr1 = "Paris"
adr2 = "Saint-Petersburg"
get_distance(adr1,adr2)

# from geopy.geocoders import Nominatim
# geolocator = Nominatim(user_agent="")
# location = geolocator.reverse("52.509669, 13.376294")
# print(location.address)
# print((location.latitude, location.longitude))
# print(location.raw)
# g = geocoder.ip('me')
# print(g.lat,g.lng)

