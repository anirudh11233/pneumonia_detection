import csv
import random
import math

# Define the Haversine function to calculate distances
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

# Function to generate a random GPS coordinate within given bounds
def generate_random_coordinates(min_lat, max_lat, min_lon, max_lon):
    accident_lat = random.uniform(min_lat, max_lat)
    accident_lon = random.uniform(min_lon, max_lon)
    return accident_lat, accident_lon

# Function to find the nearest hospital (without resources check)
def find_nearest_hospital(accident_lat, accident_lon, csv_file):
    nearest_hospital = None
    min_distance = float('inf')

    try:
        # Open the file with UTF-8 encoding to handle special characters
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Print the actual fieldnames to check if the headers match
            print(f"CSV Headers: {reader.fieldnames}")
            
            for row in reader:
                try:
                    # Convert hospital latitude and longitude to float
                    hospital_lat = float(row['Latitude'])
                    hospital_lon = float(row['Longitude'])
                    
                    # Calculate the distance from the accident to this hospital
                    distance = haversine(accident_lat, accident_lon, hospital_lat, hospital_lon)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_hospital = {
                            "Hospital Name": row['Hospital Name'],  # Updated column name
                            "Address": row['Address'],              # Updated column name
                            "City": row['City'],                    # Updated column name
                            "State": row['State'],                  # Updated column name
                            "Pin Code": row['Pin Code'],            # Updated column name
                            "Latitude": hospital_lat,
                            "Longitude": hospital_lon,
                            "Distance": min_distance
                        }
                except ValueError:
                    print(f"Skipping invalid data row: {row}")
                    continue  # Skip rows with invalid latitude/longitude
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return None
    except UnicodeDecodeError:
        print(f"Error: Unable to read the file '{csv_file}' due to encoding issues.")
        return None

    return nearest_hospital

# Define the geographic bounds (example values; adjust based on your region)
min_lat, max_lat = 13.0, 13.1  # A smaller range for latitude
min_lon, max_lon = 77.6, 77.8  # A smaller range for longitude

# Generate random accident coordinates
accident_lat, accident_lon = generate_random_coordinates(min_lat, max_lat, min_lon, max_lon)
print(f"Accident Location (Latitude, Longitude): ({accident_lat}, {accident_lon})")

# Specify the path to your CSV file
csv_file = 'C:\\Users\\saian\\Downloads\\enriched_hospitals_data.csv'

# Find the nearest hospital
nearest_hospital = find_nearest_hospital(accident_lat, accident_lon, csv_file)

if nearest_hospital:
    print("Nearest Hospital:")
    print(f"Hospital Name: {nearest_hospital['Hospital Name']}")
    print(f"Address: {nearest_hospital['Address']}")
    print(f"City: {nearest_hospital['City']}")
    print(f"State: {nearest_hospital['State']}")
    print(f"Pin Code: {nearest_hospital['Pin Code']}")
    print(f"Location: ({nearest_hospital['Latitude']}, {nearest_hospital['Longitude']})")
    print(f"Distance: {nearest_hospital['Distance']} km")
else:
    print("No hospital found.")
