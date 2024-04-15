

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.errors import ShapelyDeprecationWarning
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)

# Load the cleaned service requests data
df = pd.read_csv('C:\\Users\\ia000040\\Documents\\GreenAidProject\\Data\\Cleaned_Service_Requests.csv')

# Convert 'RECEIVED' column to datetime
df['RECEIVED'] = pd.to_datetime(df['RECEIVED'])

# Define filtering criteria
start_date = '2007-01-01'
end_date = '2011-12-31'
service_types = ["Overgrown Vegetation", "Housing - Defect", "Rats Domestic INSIDE property FREE", "Flytipping"]
wards_of_interest = ["Headingley & Hyde Park"]

# Apply filters
df_filtered = df[(df['RECEIVED'].between(start_date, end_date)) &
                 df['SR TYPE DESC'].isin(service_types) &
                 df['WARD'].isin(wards_of_interest)]

# Load geographic data (shapefile)
gdf = gpd.read_file("C:\\Users\\ia000040\\Documents\\GreenAidProjectQuarto\\GBR_adm\\GBR_adm2.shp")

# Ensure that the geographic dataframe only includes wards present in the service requests dataframe
gdf = gdf[gdf['NAME_2'].isin(df_filtered['WARD'].unique())]

# Merge the geographic and service requests dataframes on the ward column
merged_gdf = gdf.set_index('NAME_2').join(df_filtered.set_index('WARD'))

# Group data by ward and count the number of service calls
merged_gdf['service_calls'] = merged_gdf.groupby(merged_gdf.index)['SR TYPE DESC'].transform('count')

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_gdf.plot(column='service_calls', ax=ax, legend=True,
                legend_kwds={'label': "Number of Service Calls by Ward",
                             'orientation': "horizontal"})
ax.set_title('Service Calls by Ward')
plt.show()






