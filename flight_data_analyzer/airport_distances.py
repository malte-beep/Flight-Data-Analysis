"""
This module provides a function to calculate the distance between two points on the earth's surface,
given their latitude and longitude in decimal degrees.
"""
import math
import unittest



def compute_airport_distances(latitude1, longitude1, latitude2, longitude2):
    """
    Calculate the distance between two points on the earth specified in decimal degrees.
    
    Parameters:
    latitude1 (float): Latitude of the first point in decimal degrees.
    longitude1 (float): Longitude of the first point in decimal degrees.
    latitude2 (float): Latitude of the second point in decimal degrees.
    longitude2 (float): Longitude of the second point in decimal degrees.
    
    Returns:
    float: Distance between the two points in kilometers.
    """
    earth_radius_km = 6371  # Radius of the earth in kilometers
    radian_conversion_factor = math.pi / 180

    delta_lat_radians = (latitude2 - latitude1) * radian_conversion_factor
    delta_lon_radians = (longitude2 - longitude1) * radian_conversion_factor
    latitude1_radians = latitude1 * radian_conversion_factor
    latitude2_radians = latitude2 * radian_conversion_factor

    cos_delta_lat = math.cos(delta_lat_radians)
    cos_lat1_lat2 = math.cos(latitude1_radians) * math.cos(latitude2_radians)
    cos_delta_lon = math.cos(delta_lon_radians)

    haversine_formula = 0.5 - cos_delta_lat / 2 + cos_lat1_lat2 * (1 - cos_delta_lon) / 2

    distance = 2 * earth_radius_km * math.asin(math.sqrt(haversine_formula))

    return distance


# Unit tests
class TestCalculateDistance(unittest.TestCase):

    def test_same_airport(self):
        # Test distance between the same airport
        distance = compute_airport_distances(40.6413, -73.7781, 40.6413, -73.7781)
        self.assertAlmostEqual(distance, 0.0, delta=0.001)  # Delta for floating-point comparison

    def test_different_airports_same_continent(self):
        # Test distance between two airports in the same continent (New York to Los Angeles)
        distance = compute_airport_distances(40.63980, -73.77890, 33.94250, -118.4080)
        self.assertAlmostEqual(distance, 3974.01, delta=5.0)  # Approximate distance, delta for tolerance

    def test_different_airports_different_continents(self):
        # Test distance between two airports in different continents (New York to London)
        distance = compute_airport_distances(40.63980, -73.77890, 51.47060, -0.46190)
        self.assertAlmostEqual(distance, 5539.38, delta=5.0)  # Approximate distance, delta for tolerance


if __name__ == "__main__":
    unittest.main()
