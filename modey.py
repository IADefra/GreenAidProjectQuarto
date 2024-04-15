import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import GridSearchCV
from joblib import dump

# Load your data
file_path = 'C:\\Users\\ia000040\\Documents\\GreenAidProject\\Data\\Cleaned_Service_Requests.csv'
df = pd.read_csv(file_path)

# Convert RECEIVED to datetime if it isn't already
if not np.issubdtype(df['RECEIVED'].dtype, np.datetime64):
    df['RECEIVED'] = pd.to_datetime(df['RECEIVED'])

# Extract year and month from RECEIVED column
df['YEAR'] = df['RECEIVED'].dt.year
df['MONTH'] = df['RECEIVED'].dt.month

# Aggregate data
grouped_df = df.groupby(['YEAR', 'MONTH', 'SR TYPE DESC', 'WARD']).size().reset_index(name='COUNT')



# Specify wards and service types of interest
wards_of_interest = ["Headingley & Hyde Park", "Harewood", "Temple Newsam", "Middleton Park"]
service_types = ["Overgrown Vegetation", "Housing - Defect", "Rats Domestic INSIDE property  FREE", "Flytipping"]

# Filter the dataframe for the wards and service request types of interest
filtered_df = grouped_df[
    grouped_df['WARD'].isin(wards_of_interest) & grouped_df['SR TYPE DESC'].isin(service_types)
]



# Outlier Detection and Removal
Q1 = filtered_df['COUNT'].quantile(0.25)
Q3 = filtered_df['COUNT'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
filtered_df = filtered_df[(filtered_df['COUNT'] >= lower_bound) & (filtered_df['COUNT'] <= upper_bound)]


# One-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(df[['SR TYPE DESC', 'WARD']])

# Perform the encoding for the filtered_df
X_encoded = ohe.transform(filtered_df[['SR TYPE DESC', 'WARD']])
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=ohe.get_feature_names_out())

# Prepare features and target
X = pd.concat([filtered_df[['YEAR', 'MONTH']].reset_index(drop=True), X_encoded_df], axis=1)
y = filtered_df['COUNT']

# Ensure X and y have the same length
assert len(X) == len(y), "Length of X and y must be equal"

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)



# Train the model with adjusted parameters
from sklearn.ensemble import GradientBoostingRegressor

# Train the Gradient Boosting model
gbm_model = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages to be run
    learning_rate=0.1,  # Shrinks the contribution of each tree by learning_rate
    max_depth=3,  # Maximum depth of the individual regression estimators
    min_samples_split=5,  # The minimum number of samples required to split an internal node
    min_samples_leaf=1,  # The minimum number of samples required to be at a leaf node
    random_state=42  # Ensures reproducibility
)
gbm_model.fit(X_train, y_train)

# Model evaluation
y_pred_gbm = gbm_model.predict(X_test)
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gbm)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_gbm)}')
print(f'R-squared: {r2_score(y_test, y_pred_gbm)}')


# Future predictions
future_years = [2024]
future_months = range(1, 13)

# Generate all combinations for future predictions
future_combinations = list(product(future_years, future_months, wards_of_interest, service_types))
future_df = pd.DataFrame(future_combinations, columns=['YEAR', 'MONTH', 'WARD', 'SR TYPE DESC'])

# Encode future data using OneHotEncoder trained earlier
future_encoded = ohe.transform(future_df[['SR TYPE DESC', 'WARD']])
future_encoded_df = pd.DataFrame(future_encoded.toarray(), columns=ohe.get_feature_names_out())

# Prepare features for prediction
future_X = pd.concat([future_df[['YEAR', 'MONTH']], future_encoded_df], axis=1)

# Predict future service request counts
future_y = gbm_model.predict(future_X)
future_df['PREDICTED_COUNT'] = future_y

# Specify the directory and filename for the saved model
model_directory = 'C:\\Users\\ia000040\\Documents\\GreenAidProjectQuarto\\model'
model_filename = 'gbm_model.joblib'
model_path = f"{model_directory}\\{model_filename}"

# Ensure the directory exists
import os
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save the model
dump(gbm_model, model_path)

print(f"Model successfully saved at {model_path}")




# Visualization with vertical grid lines for each month in the forecasted period
for ward in wards_of_interest:
    plt.figure(figsize=(15, 5))
    ward_data = future_df[future_df['WARD'] == ward]
    ax = plt.gca()

    # Initialize a list to collect all dates across service types
    all_dates_list = []

    for service_type in service_types:
        st_data = ward_data[ward_data['SR TYPE DESC'] == service_type]
        # Combine YEAR and MONTH into a single date string
        dates = st_data['YEAR'].astype(str) + '-' + st_data['MONTH'].astype(str).str.zfill(2)
        # Plot the data
        plt.plot(dates, st_data['PREDICTED_COUNT'], label=service_type)
        # Collect the dates
        all_dates_list.extend(dates.tolist())

    # Find unique dates and sort them
    unique_dates = sorted(set(all_dates_list))
    # Create a range for x-ticks
    x_ticks_range = range(len(unique_dates))

    # Set the x-ticks and labels
    ax.set_xticks(x_ticks_range)
    ax.set_xticklabels(unique_dates, rotation=45, ha='right')

    plt.grid(which='major', axis='x', linestyle='--', color='black')
    plt.title(f'Forecasted Service Requests in {ward}')
    plt.xlabel('Year-Month')
    plt.ylabel('Predicted Requests')
    plt.legend()
    plt.tight_layout()
    plt.show()


