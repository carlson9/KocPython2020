from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent = 'dcarlson@ku.edu.tr') #use your email address
location = geolocator.geocode('Washington, DC')
print(location.address)
print((location.latitude, location.longitude))

location2 = geolocator.geocode('Mexico City')
print(location2.address)
print((location2.latitude, location2.longitude))

#distance between capitals
from math import radians, sin, cos, acos

def distance(loc1, loc2):
	return 6371.01 * acos(sin(radians(loc1.latitude))*sin(radians(loc2.latitude)) + cos(radians(loc1.latitude))*cos(radians(loc2.latitude))*cos(radians(loc1.longitude) - radians(loc2.longitude)))
	
distance(location, location2)

#TODO: Create a distance matrix of 5 capitals

locations = []
locations.append(geolocator.geocode('Washington, DC'))
locations.append(geolocator.geocode('Mexico City'))
locations.append(geolocator.geocode('Sarajevo'))
locations.append(geolocator.geocode('Ankara'))
locations.append(geolocator.geocode('Beijing'))

Matrix = [[0 for x in range(5)] for y in range(5)]

for x in range(5):
	for y in range(5):
		Matrix[x][y] = distance(locations[x], locations[y])
		

