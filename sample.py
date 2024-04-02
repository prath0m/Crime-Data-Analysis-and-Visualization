# Import the necessary module
from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")

# Define the latitude and longitude
latitude = 18.5204
longitude = 73.8567

# Get the location information
location = geolocator.reverse((latitude, longitude))

# Extract the pincode from the location
pincode = location.raw['address']['postcode']
print("Pincode:", pincode)