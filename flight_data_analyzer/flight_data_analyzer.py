# pylint: disable=E1136
"""
This module defines the FlightDataAnalyzer class, which is designed to download,
extract, and analyze flight data from a specified source. The analysis includes
computing distances between airports and visualizing flight routes.
"""

import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from openai import OpenAI
import airport_distances



class FlightDataAnalyzer:
    """
    A class for analyzing and visualizing flight data. It includes methods for downloading
    flight data from a specified source, extracting and loading it into data structures,
    and performing various analyses such as distance calculations and plotting flight routes
    and airport locations using geographical data.

    Attributes:
        file_url (str): URL of the zip file containing flight data to download.
        downloads_path (str): Local directory path where the zip file
        and its extracted contents will be saved.
        zip_file_path (str): Full path to the downloaded zip file.
        data (dict): A dictionary holding pandas DataFrames loaded from the extracted CSV files.

    Methods include capabilities to download and extract data, load CSV files into DataFrames,
    plot airports by country, analyze flight distances, plot flights by airport, identify the most
    used airplane models, and plot flights by country, with options to filter by internal flights.
    """
    def __init__(self,
                 file_url='https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip',
                 downloads_path='downloads', zip_file_name='flight_data.zip'):
        self.file_url = file_url
        self.downloads_path = downloads_path
        self.zip_file_path = os.path.join(self.downloads_path, zip_file_name)
        self._ensure_data_file_exists_and_extract()
        self.data = self.load_csv_to_dataframe()
        self.client = OpenAI()

    def _ensure_data_file_exists_and_extract(self):
        if not os.path.exists(self.zip_file_path):
            self._download_file_from_gitlab()
        self._extract_zip_file()

    def _download_file_from_gitlab(self):
        print(f"Attempting to download {self.file_url}...")
        try:
            response = requests.get(self.file_url, timeout=10)  # 10 seconds timeout
            if response.status_code == 200:
                os.makedirs(self.downloads_path, exist_ok=True)
                with open(self.zip_file_path, 'wb') as file_handle:
                    file_handle.write(response.content)
                print(f"File downloaded successfully: {self.zip_file_path}")
            else:
                print(f"Failed to download file. Status code: {response.status_code}")
        except requests.exceptions.Timeout:
            print("The request timed out. Please try again later.")
        except requests.exceptions.RequestException as error:
            print(f"An error occurred while trying to download the file: {error}")

    def _extract_zip_file(self):
        """
        Extracting the 4 csv files from the zip file
        """
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.downloads_path)
        print(f"Files extracted to {self.downloads_path}")

    def load_csv_to_dataframe(self):
        """
        Dictionary to map file names to DataFrame attribute names
        """
        file_to_attr = {
        'airlines.csv': 'airlines',
        'airplanes.csv': 'airplanes',
        'airports.csv': 'airports',
        'routes.csv': 'routes',
        }

        df_dict = {
        'airlines': None,
        'airplanes': None,
        'airports': None,
        'routes': None,
        }

        for file_name, attr_name in file_to_attr.items():
            df_path = os.path.join(self.downloads_path, file_name)
            df_dict[attr_name] = pd.read_csv(df_path, index_col= "index")

        df_dict["airlines"] = df_dict["airlines"].iloc[1:].reset_index(drop=True)


        # Rename some country names to match the world map
        matches = {'Western Sahara': 'W. Sahara',
                   'United States': 'United States of America',
                   'Congo (Kinshasa)': 'Dem. Rep. Congo',
                   'Dominican Republic': 'Dominican Rep.',
                   'Falkland Islands': 'Falkland Is.',
                   'French Southern and Antarctic Lands': 'Fr. S. Antarctic Lands',
                   'East Timor': 'Timor-Leste',
                   "Cote d'Ivoire": "Côte d'Ivoire",
                   'Central African Republic': 'Central African Rep.',
                   'Congo (Brazzaville)': 'Congo',
                   'Equatorial Guinea': 'Eq. Guinea',
                   'Swaziland': 'eSwatini',
                   'Solomon Islands': 'Solomon Is.',
                   'Czech Republic': 'Czechia',
                   'Northern Cyprus': 'N. Cyprus',
                   'Somaliland': 'Somaliland',  # No match found in the first list
                   'Bosnia and Herzegovina': 'Bosnia and Herz.',
                   'Macedonia': 'North Macedonia',  # No exact match found,
                   # using 'North Macedonia' from first list
                   'Kosovo': 'Kosovo',  # No match found in the first list
                   'South Sudan': 'S. Sudan'
}

        # Rename observations in the DataFrame
        # If there is no match, the original name is kept
        df_dict["airports"]["Country"] = df_dict["airports"]["Country"].replace(matches)

        return df_dict

    def adapt_lists(self):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        countries = self.data["airports"]["Country"].unique()

        # Show countries that are not in the world map
        not_mapped_countries = []
        for country in countries:
            if country not in world["name"].to_list():
                not_mapped_countries.append(country)

        # Show world map countries that are not in the airports data
        not_mapped_world = []
        for country in world["name"].to_list():
            if country not in countries:
                not_mapped_world.append(country)

        return not_mapped_countries, not_mapped_world

    def plot_country_airports(self, country_name: str):
        """
        Plot all Airports for a given Country using GeoPandas and return a pyplot object

        Parameters:
            - country_name: The name of the country for which airport locations should be plotted
        """

        # Load the world map using GeoPandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        if country_name not in world["name"].to_list():
            raise ValueError("The provided name does not belong to any country")

        airports_df = self.data['airports']
        country_airports = airports_df[airports_df['Country'] == country_name]

        # Convert longitude and latitude to Points
        geometry = [Point(xy) for xy in zip(country_airports['Longitude'],
                                            country_airports['Latitude'])]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(country_airports, geometry=geometry)

        # Set the coordinate reference system (CRS) to WGS84 (standard for GPS data)
        gdf.crs = "EPSG:4326"

        # Plot the world map
        base = world[world.name == country_name].plot(color='white',
                                                      edgecolor='black', figsize=(10, 10))

        # Plot airports on top of the world map
        gdf.plot(ax=base, marker='o', color='red', markersize=5)

        # Add title and show the plot
        plt.title(f'Airports in {country_name}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        return plt

    def distance_analysis(self):
        """
        Analyzes and plots the distribution of distances for all flight routes. It calculates
        the distance between source and destination airports and visualizes the results in
        a histogram.

        Returns:
        - matplotlib.pyplot: The plot object showing the distribution of flight distances.
        """
        airports_loc = self.data["routes"].merge(self.data["airports"][["IATA",
                                                                        "Latitude", "Longitude"]],
                                                 left_on="Source airport",
                                                 right_on="IATA")

        airports_loc.drop(columns=["IATA"], inplace=True)

        airports_loc.rename(columns={"Latitude": "Source airport Lat",
                                     "Longitude": "Source airport Lon"}, inplace=True)

        airports_loc = airports_loc.merge(self.data["airports"][["IATA", "Latitude", "Longitude"]],
                                          left_on="Destination airport",
                                          right_on="IATA")

        airports_loc.drop(columns=["IATA"], inplace=True)

        airports_loc.rename(columns={"Latitude": "Destination airport Lat",
                                     "Longitude": "Destination airport Lon"},
                                     inplace=True)

        airports_loc["Distance"] = airports_loc.apply(
            lambda x: airport_distances.compute_airport_distances(
                x.loc["Source airport Lat"],
                x.loc["Source airport Lon"],
                x.loc["Destination airport Lat"],
                x.loc["Destination airport Lon"]
            ), axis=1
        )

        plt.hist(airports_loc["Distance"])
        plt.title("Distribution of Flight Distances in Kilometers (km)")
        plt.xlabel("Flight Distance")
        plt.ylabel("Amount of Routes")

        return plt

    def plot_flights_by_airport(self, airport: str, internal = False):
        """
        Plots flight routes originating from a specified airport. Optionally, it can filter
        to display only internal routes within the same country as the airport. The function
        merges route data with airport information to obtain geographical coordinates and
        country data, then uses GeoPandas and matplotlib to visualize the routes and airports
        on a world map.

        Parameters:
        - airport (str): IATA code of the source airport from which routes will be plotted.
        - internal (bool, optional): If True, only routes that have both their source and
        destination within the same country as the source airport are plotted. If False,
        all routes from the source airport are plotted. Defaults to False.

        Returns:
        - matplotlib.pyplot: The plot object, which can be displayed using plt.show() or
        saved with plt.savefig().
        """
        airports = self.data["airports"]
        routes = self.data["routes"]
        routes = routes.loc[routes["Source airport"] == airport]
        country = airports.loc[airports["IATA"] == airport, "Country"].iloc[0]

        airports_reduced = airports[["Country", "IATA", "Latitude", "Longitude"]]

        routes = routes.merge(airports_reduced, left_on = "Source airport",
                              right_on = "IATA").drop(columns="IATA")
        routes.rename(columns = {"Country": "Source country",
                                 "Latitude": "Source airport lat",
                                 "Longitude": "Source airport lon",},
                      inplace = True)

        routes = routes.merge(airports_reduced, left_on="Destination airport",
                              right_on="IATA").drop(columns="IATA")
        routes.rename(columns={"Country": "Destination country",
                               "Latitude": "Destination airport lat",
                               "Longitude": "Destination airport lon", },
                      inplace=True)

        if internal:
            routes = routes.loc[routes["Destination country"] == country]


        # Convert longitude and latitude to Points
        source_point = [Point(xy) for xy in zip(routes['Source airport lon'],
                                                routes['Source airport lat'])]
        destination_point = [Point(xy) for xy in zip(routes['Destination airport lon'],
                                                     routes['Destination airport lat'])]

        geometry = [LineString([source, destination]) for source,
                    destination in zip(source_point, destination_point)]


        # Create a GeoDataFrame
        gdf_lines = gpd.GeoDataFrame(routes, geometry=geometry)



        origins = routes[["Source airport", "Source airport lat",
                          "Source airport lon"]]
        origins = origins.rename(columns = {"Source airport" : "IATA",
                                            "Source airport lat" : "Latitude",
                                            "Source airport lon" : "Longitude"})
        destinations = routes[["Destination airport", "Destination airport lat",
                               "Destination airport lon"]]
        destinations = destinations.rename(columns={"Destination airport": "IATA",
                                                    "Destination airport lat": "Latitude",
                                                    "Destination airport lon": "Longitude"})
        airport_points = pd.concat([origins, destinations]).drop_duplicates()
        geometry = [Point(xy) for xy in zip(airport_points['Longitude'],
                                            airport_points['Latitude'])]
        gdf_point = gpd.GeoDataFrame(airport_points, geometry=geometry)


        # Load the world map using GeoPandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # Set the coordinate reference system (CRS) to WGS84 (standard for GPS data)
        gdf_lines.crs = "EPSG:4326"
        gdf_point.crs = "EPSG:4326"
        world = world.to_crs("EPSG:4326")

        # Plot the world map
        if internal:
            base = world[world.name == country].plot(color='white',
                                                     edgecolor='black',
                                                     figsize=(10, 10))
        else:
            base = world.plot(color='white', edgecolor='black', figsize=(10, 10))

        # Plot airports on top of the world map
        gdf_lines.plot(ax=base, marker='--', color='#009e73', linewidth = 1, alpha = 0.4)
        gdf_point.plot(ax=base, marker='o', color='red', markersize=5)


        # Add title and show the plot
        plt.title(f'Routes from {airport}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        return plt


    def most_airplane_model(self, countries = None, print_n = 10):
        """
        Identifies and plots the most used airplane models,
        based on the provided flight routes data.
        This can be filtered for specific countries
        and can display a specified number of top models.

        Parameters:
        - countries (None, str, or list of str, optional):
        A country or list of countries to filter the routes by.
        If None, all routes are considered. Defaults to None.
        - print_n (int, optional): The number of top airplane models to display in the plot.
        Defaults to 10.

        Returns:
        - matplotlib.pyplot: The plot object with the most used airplane models.
        This can be shown using plt.show() or saved to a file with plt.savefig().
        """
        routes = self.data["routes"]
        airports = self.data["airports"]

        if countries is not None:
            if isinstance(countries, str):
                countries = [countries]

            airports = airports.loc[airports["Country"].isin(countries)]
            routes = routes.loc[routes["Source airport"].isin(airports["IATA"])]


        routes = routes[["Source airport", "Destination airport", "Equipment"]].drop_duplicates()

        routes["Equipment"] = routes["Equipment"].str.split(" ")
        routes = routes.explode("Equipment")
        equipment_count = routes.groupby("Equipment").size()
        sorted_equipment = equipment_count.sort_values(ascending=False)
        top_n_equipment = sorted_equipment.head(print_n)

        # Plot the results
        top_n_equipment.plot(kind="bar")
        plt.title("Most used airplane models")
        plt.ylabel("Amount of routes")
        plt.xlabel("Airplane model")

        return plt

    def plot_flights_by_country(self, country: str, internal=False, cutoff_distance: float = None):
        """
        Plots flight routes and airports for a specified country, differentiating
        between short-haul and long-haul flights based on a cutoff distance. It
        merges route and airport data, filters by the given country, and plots both
        the flight routes and airport locations on a map. The map can display either
        all global routes from and to the specified country or only the internal
        routes within the country.

        Additionally, annotates the plot with the number of short-haul routes and
        the total distance covered by these routes, avoiding double counting.

        Parameters:
        - country (str): The country for which flight routes and airports are to
          be plotted. The country name should match the names used in the airports
          data.
        - internal (bool, optional): If True, only internal flight routes
          (i.e., both source and destination airports are within the specified
          country) are plotted. Defaults to False.
        - cutoff_distance (float, optional): The distance (in kilometers) below
          which a flight is considered short-haul. If not specified, all flights
          are considered without categorization.

        Returns:
        - matplotlib.pyplot: The plot object with flight routes and airports
          plotted. This can be shown with plt.show() or saved to a file using
          plt.savefig().
        """
        airports = self.data["airports"]
        routes = self.data["routes"]

        # Merge with airport data for source airports
        merged_routes = routes.merge(
            airports.loc[airports["Country"] == country][["IATA", "Latitude", "Longitude"]],
            left_on="Source airport", right_on="IATA").drop(columns="IATA")
        merged_routes.rename(columns={"Latitude": "Source airport lat",
                                      "Longitude": "Source airport lon"},
                             inplace=True)

        if internal:
            airports = airports.loc[airports["Country"] == country]

        # Merge with airport data for destination airports
        merged_routes = merged_routes.merge(
            airports[["IATA", "Latitude", "Longitude"]],
            left_on="Destination airport", right_on="IATA").drop(columns="IATA")
        merged_routes.rename(columns={"Latitude": "Destination airport lat",
                                      "Longitude": "Destination airport lon"},
                             inplace=True)

        # Calculate distances and categorize flights
        merged_routes['Distance'] = merged_routes.apply(
            lambda row: airport_distances.compute_airport_distances(
                row['Source airport lat'], row['Source airport lon'],
                row['Destination airport lat'], row['Destination airport lon']), axis=1)

        if cutoff_distance is not None:
            merged_routes['Flight Type'] = merged_routes['Distance'].apply(
                lambda x: 'Short-haul' if x <= cutoff_distance else 'Long-haul')

            # Calculate unique short-haul distances before emissions calculations
            short_haul = merged_routes[merged_routes['Flight Type'] == 'Short-haul'].copy()

            short_haul['Route'] = short_haul.apply(
                lambda x: '-'.join(sorted([x['Source airport'], x['Destination airport']])), axis=1)
            unique_short_haul = short_haul.drop_duplicates(subset=['Route'])

            total_distance_short_haul_km = unique_short_haul['Distance'].sum()

        # Prepare geometries
        geometry = [LineString([(row['Source airport lon'], row['Source airport lat']),
                                (row['Destination airport lon'], row['Destination airport lat'])])
                    for index, row in merged_routes.iterrows()]

        gdf_lines = gpd.GeoDataFrame(merged_routes, geometry=geometry)

        # Set the CRS
        gdf_lines.crs = "EPSG:4326"

        # Load the world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs("EPSG:4326")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10)) # pylint: disable=invalid-name

        # Define country_geom outside the if-else structure to ensure it covers both cases
        country_geom = world[world['name'] == country].geometry

        # Plot the country outlines
        country_geom.boundary.plot(ax=ax, linewidth=1,
                                   edgecolor='black',
                                   label=f'{country} Boundary')

        if internal:
            # Zoom to the country of interest for internal flights
            country_geom = world[world['name'] == country]
        else:
            world.plot(ax=ax, color='white', edgecolor='black')

        # Plot flights with color coding for short and long-haul, if applicable
        if cutoff_distance is not None:
            if gdf_lines.loc[gdf_lines['Flight Type'] == 'Long-haul', 'Flight Type'].empty is False:
                gdf_lines[gdf_lines['Flight Type'] == 'Long-haul'].plot(
                    ax=ax, color='#d55e00', linewidth=1, label='Long-haul', alpha=0.4)
            gdf_lines[gdf_lines['Flight Type'] == 'Short-haul'].plot(
                ax=ax, color='#0072b2', linewidth=1, label='Short-haul', alpha=0.4)

            # Constants for emissions (Kg CO2-eq per passenger km for Australia)
            flight_emission_factor = 0.151  # Kg CO2-eq per passenger km for short-haul flights
            rail_emission_factor = 0.00011   # Kg CO2-eq per passenger km for rail

            # Calculate total emissions from short-haul flights
            flight_emissions = total_distance_short_haul_km * flight_emission_factor

            # Calculate equivalent emissions if replaced by rail
            rail_emissions = total_distance_short_haul_km * rail_emission_factor

            # Calculate emissions reduction
            emissions_reduction = flight_emissions - rail_emissions


            # Depending on whether the plot is for internal flights or not,
            # the annotations are placed in different positions
            if internal:
                # Annotate plot with emissions reduction information
                amount_distance_short_haul = (
                    f"Short-haul routes: {unique_short_haul.shape[0]}\n"
                    f"Total short-haul distance: {total_distance_short_haul_km:.2f} Km")
                plt.annotate(amount_distance_short_haul, xy=(0, -0.08), xycoords='axes fraction',
                             ha='left', va='top', fontsize=8)
                emissions_reduction_annotation = (
                    f"Emissions reduction by replacing short-haul flights with rail: "
                    f"{emissions_reduction:.2f} Kg CO2-eq")
                plt.annotate(emissions_reduction_annotation,
                             xy=(0, -0.115), xycoords='axes fraction',
                             ha='left', va='top', fontsize=8)
            else:
                # Annotate plot with emissions reduction information
                amount_distance_short_haul = (
                    f"Short-haul routes: {unique_short_haul.shape[0]}\n"
                    f"Total short-haul distance: {total_distance_short_haul_km:.2f} Km")
                plt.annotate(amount_distance_short_haul, xy=(0.03, -0.08), xycoords='axes fraction',
                             backgroundcolor = "white" , ha='left', va='top', fontsize=8)
                emissions_reduction_annotation = (
                    f"Emissions reduction by replacing short-haul flights with rail: "
                    f"{emissions_reduction:.2f} Kg CO2-eq")
                plt.annotate(emissions_reduction_annotation, xy=(0.03, -0.155),
                             xycoords='axes fraction',
                             backgroundcolor = "white" ,ha='left', va='top', fontsize=8)


        else:
            gdf_lines.plot(ax=ax, color='#009e73', linewidth=1, alpha=0.4)

        # Plot airports with routes using red dots
        # Get unique IATA codes of airports involved in routes
        source_airports = set(merged_routes['Source airport'])
        destination_airports = set(merged_routes['Destination airport'])
        active_airports = source_airports.union(destination_airports)

        # Filter the airports DataFrame to only include these airports
        active_airports_df = airports[airports['IATA'].isin(active_airports)]

        # Now, create the GeoDataFrame for these active airports only
        airport_points = [Point(xy) for xy in zip(active_airports_df['Longitude'],
                                                      active_airports_df['Latitude'])]
        gdf_active_airports = gpd.GeoDataFrame(active_airports_df, geometry=airport_points)
        gdf_active_airports.crs = "EPSG:4326"

        # Plot these active airports as dots on the map
        gdf_active_airports.plot(ax=ax, marker='o', color='red', markersize=5,
                                     label='Active Airports')

        plt.legend()
        plt.title(f'Flight Routes from {country} (Short-haul vs Long-haul)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Adjust plot limits for internal flights
        if internal:
            minx, miny, maxx, maxy = country_geom.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        return plt

    def aircrafts(self):
        """
        Prints the list of all aircraft names from the dataset.

        This method accesses the 'airplanes' dataset contained within the 'data' attribute,
        extracts the 'Name' column, and prints a list of all aircraft names.
        """
        print(self.data["airplanes"]["Name"].to_list())

    def aircraft_info(self, aircraft_name: str):
        """
        Retrieves and returns information about an aircraft by its name, IATA code, or ICAO code.

        This method searches the aircraft dataset for a matching entry
        based on the provided name or code. If a match is found,
        it fetches detailed information using an external API (assumed to be OpenAI's GPT model)
        and returns a generated text containing the aircraft information.

        Parameters:
        - aircraft_name (str): The name, IATA code, or ICAO code of the aircraft.

        Returns:
        - str: Detailed information about the aircraft as a string.

        Raises:
        - ValueError: If the provided name does not match any aircraft in the dataset.
        """
        aircrafts = self.data["airplanes"]
        client = self.client

        if aircraft_name in aircrafts["Name"].to_list():
            aircraft_name_ret = aircrafts.loc[
                aircrafts["Name"] == aircraft_name, "Name"].iloc[0]
        elif aircraft_name in aircrafts["IATA code"].to_list():
            aircraft_name_ret = aircrafts.loc[
                aircrafts["IATA code"] == aircraft_name, "Name"].iloc[0]
        elif aircraft_name in aircrafts["ICAO code"].to_list():
            aircraft_name_ret = aircrafts.loc[
                aircrafts["ICAO code"] == aircraft_name, "Name"].iloc[0]
        else:
            raise ValueError(
                "The provided name does not belong to any aircraft."
                "Use the aircrafts() method to see the list of available aircrafts.")

        with open("flight_data_analyzer/prompt_aircraft.txt", encoding='utf-8') as prompt_file:
            prompt = prompt_file.read().replace("AIRCRAFT_NAME_RET", aircraft_name_ret)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        result = completion.choices[0].message.content

        return result

    def airports(self):
        """
        Prints a list of cities from the airports dataset.
        """
        print(self.data["airports"]["City"].to_list())

    def airport_info(self, airport_name: str):
        """
        Retrieves information for a specified airport by its city name or IATA/ICAO code.

        Parameters:
        airport_name: The city name, IATA, or ICAO code of the airport.

        Returns:
        The retrieved airport information as a string.
        """
        airports = self.data["airports"]
        client = self.client

        if airport_name in airports["City"].str.strip().to_list():
            airport_name_ret = airports.loc[airports["City"] == airport_name, "City"].iloc[0]
        elif airport_name in airports["IATA"].to_list():
            airport_name_ret = airports.loc[airports["IATA"] == airport_name, "Name"].iloc[0]
        elif airport_name in airports["ICAO"].to_list():
            airport_name_ret = airports.loc[airports["ICAO"] == airport_name, "Name"].iloc[0]
        else:
            raise ValueError(
                "The provided city name or ICAO/IATA does not belong to any airport."
                "Use the airports() method to see the list of available airports.")

        with open("flight_data_analyzer/prompt_airport.txt", encoding='utf-8') as prompt_file:
            prompt = prompt_file.read().replace("AIRPORT_NAME_RET", airport_name_ret)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        result = completion.choices[0].message.content

        return result
