import streamlit as st
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import openpyxl
import matplotlib.pyplot as plt

#######################################
# PAGE SETUP
#######################################


st.set_page_config(layout="wide", page_title="Financial Dashboard", page_icon=":bar_chart:", initial_sidebar_state="expanded")

primary_color = "#E50E4F"
st.markdown(

f"""

<style>

.reportview-container .main .block-container{{

max-width: 1200px;

padding-top: 5rem;

padding-right: 2rem;

padding-left: 2rem;

padding-bottom: 5rem;

}}

.sidebar .sidebar-content {{

background-color: #0E1117;

}}

header .decoration {{

background-color: {primary_color};

}}

.css-1d391kg, .st-bb {{

color: {primary_color};

}}

.st-at {{

background-color: {primary_color};

}}

</style>

""",

unsafe_allow_html=True

#
)





#######################################
# DATA LOADING


# Set page config to use a wide layout and a dark theme
#st.set_page_config(layout="wide", page_title="Financial Dashboard", theme="dark")

# Define the primary color for accents based on the image
primary_color = "#E50E4F"

# Custom CSS to inject for fonts and additional styles
# st.markdown(
#     f"""
#     <style>
#     .reportview-container .main .block-container{{
#         max-width: 1200px;
#         padding-top: 5rem;
#         padding-right: 2rem;
#         padding-left: 2rem;
#         padding-bottom: 5rem;
#     }}
#     .sidebar .sidebar-content {{
#         background-color: #0E1117;
#     }}
#     header .decoration {{
#         background-color: {primary_color};
#     }}
#     .css-1d391kg, .st-bb {{
#         color: {primary_color};
#     }}
#     .st-at {{
#         background-color: {primary_color};
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#)

# Sidebar for input variables

# Convert Invoice date to datetime and extract year and month
def process_data(df):
    df['Invoice date'] = pd.to_datetime(df['Invoice date'], errors='coerce')
    df['Year'] = df['Invoice date'].dt.year
    df['Month'] = df['Invoice date'].dt.month
    df['Month Name'] = df['Invoice date'].dt.strftime('%B')
    return(df)

st.title("Purchasing Dashboard")

with st.sidebar:
    data = pd.read_csv('/Users/otb/Desktop/Purchasing Project/purchasing_data_sample_csv.csv')
    if process_data is None:
        st.info("Please upload a file through the uploader.")
        st.stop()


processed_data = process_data(data)


# Function to create a data table for the input variables
def create_data_table(invoice_date, invoice_amount, po_amount, po_number, purchase_description, vendor_name, vendor_type):
    # Create a DataFrame to hold the input data
    data = {
        'Invoice date': [invoice_date],
        'Invoice amount': [invoice_amount],
        'Purchase order total': [po_amount],
        'Purchase order number': [po_number],
        'Purchase description': [purchase_description],
        'Vendor name': [vendor_name],
        'Vendor type': [vendor_type]
    }

    df = pd.DataFrame(data)
    return df


st.sidebar.header("Filters")

selected_vendor_type = st.sidebar.selectbox('Vendor Type', options=processed_data['Vendor Type description'].unique())


amount_range = st.sidebar.slider('Invoice Amount Range', 
                                min_value=float(processed_data['Invoice amount'].min()), 
                                max_value=float(processed_data['Invoice amount'].max()), 
                                value=(float(processed_data['Invoice amount'].min()), 
                                        float(processed_data['Invoice amount'].max())))

po_amount_range = st.sidebar.slider('Purchase Order Range', 
                                min_value=float(processed_data['Purchase order total'].min()), 
                                max_value=float(processed_data['Purchase order total'].max()), 
                                value=(float(processed_data['Purchase order total'].min()), 
                                        float(processed_data['Purchase order total'].max())))


# Slider for Invoice Date Range
start_date, end_date = st.sidebar.slider("Select Date Range",
                                        min_value=processed_data['Invoice date'].min().date(),
                                        max_value=processed_data['Invoice date'].max().date(),
                                        value=(processed_data['Invoice date'].min().date(), 
                                                processed_data['Invoice date'].max().date()))

# # Slider for Purchase Order Date Range
# start_date, end_date = st.sidebar.slider("Select Date Range",
#                                         min_value=processed_data['PO date'].min().date(),
#                                         max_value=processed_data['PO date'].max().date(),
#                                         value=(processed_data['PO date'].min().date(), 
#                                                 processed_data['PO date'].max().date()))

# # Combined Filtering based on selected vendor type, vendor name, date range, and amount range
# filtered_data = processed_data[(processed_data['Vendor Type description'].isin([selected_vendor_type])) &
#                             (processed_data['Vendor name'].isin(selected_vendor_name)) &
#                             (processed_data['Invoice date'].dt.date >= start_date) & 
#                             (processed_data['Invoice date'].dt.date <= end_date) &
#                             (processed_data['Invoice amount'] >= amount_range[0]) & 
#                             (processed_data['Invoice amount'] <= amount_range[1])]

with st.sidebar:
    st.header('Search')
    invoice_amount = st.number_input('Invoice amount', min_value=0.0, format='%f')
    invoice_date = st.date_input('Invoice Date')
    purchase_order_amount = st.number_input("PO amount")
    purchase_order_number = st.number_input('PO number')
    purchase_description = st.text_input('Purchase Description')
    vendor_name = st.text_input('Vendor Name')
    vendor_type = st.sidebar.selectbox('Vendor type description', options=processed_data['Vendor type description'].unique())


# Main content area
with st.container():
    st.title("Financial Dashboard")

    # Placeholder for a chart
    st.subheader("Invoice Purchase order totals Chart")
    chart_placeholder = st.empty()  # You can use chart_placeholder to display charts

    # Placeholder for a map
    st.subheader("Vendor Locations Map")
    map_placeholder = st.empty()  # You can use map_placeholder to display maps

    # Placeholder for a data table
    st.subheader("Invoices Data Table")
    table_placeholder = st.empty()  # You can use table_placeholder to display tables

    # Placeholder for KPIs
    st.subheader("Key Performance Indicators")
    kpi_placeholder = st.empty()  # You can use kpi_placeholder to display KPIs

# You would need to add the logic to generate and display the charts, maps, tables, and KPIs based on the input variables
    
marker_cluster_import_statement = 'from folium.plugins import MarkerCluster'

# Main content placeholders
st.header('Dashboard')
chart_placeholder = st.empty()  # Placeholder for charts
map_placeholder = st.empty()  # Placeholder for maps
table_placeholder = st.empty()  # Placeholder for tables
kpi_placeholder = st.empty()  # Placeholder for KPIs



# Function to create a bar chart for invoice and Purchase order totals
def create_bar_chart(invoice_amount, purchase_order_amount):
    fig, ax = plt.subplots()
    ax.bar(['Invoice amount', 'Purchase order total'], [invoice_amount, purchase_order_amount])
    ax.set_ylabel('Amount')
    ax.set_title('Invoice vs Purchase order total')
    return fig



# Main content update based on input
with st.container():
    st.header('Financial Visuals')
    
# Assuming the user has input the data, we generate the bar chart
if invoice_amount and purchase_order_amount:
    bar_chart_fig = create_bar_chart(invoice_amount, purchase_order_amount)
    st.pyplot(bar_chart_fig)




# Function to calculate and display KPIs
def display_kpis(invoice_amount, purchase_order_amount):
    total_amount = invoice_amount + purchase_order_amount
    st.metric(label="Total Amount", value=f"${total_amount:,.2f}")

# Assuming the user has input the data, we generate the data table and KPIs
if invoice_amount and purchase_order_amount and invoice_date and purchase_description and vendor_name and vendor_type:
    data_table = create_data_table(invoice_date, invoice_amount, purchase_order_amount, purchase_description, vendor_name, vendor_type)
    table_placeholder.table(data_table)
    
    display_kpis(invoice_amount, purchase_order_amount)


# # Function to create a map visualization for vendor locations
# def create_map(vendor_data):
#     # Starting coordinates for the map (this would be dynamic based on your data)
#     start_coords = (37.7749, -122.4194)  # San Francisco coordinates
#     #vendor_map = folium.Map(location=start_coords, zoom_start=12)
#     marker_cluster = MarkerCluster().add_to(vendor_map)
    
#     # # Assuming vendor_data is a DataFrame with 'latitude' and 'longitude' columns
#     # for idx, row in vendor_data.iterrows():
#     #     #folium.Marker(
#     #         location=[row['latitude'], row['longitude']],
#     #         popup=row['vendor_name']
#     #     ).add_to(marker_cluster)
    
#     return vendor_map

# # Placeholder for the map visualization
# map_placeholder = st.empty()

# Assuming we have vendor data with latitude and longitude
# Here you would load your actual data
# vendor_data = pd.DataFrame({
#     'vendor_name': ['Vendor A', 'Vendor B'],
#     'latitude': [37.7749, 37.7799],
#     'longitude': [-122.4194, -122.4144]
# })

# # Generate and display the map
# vendor_map = create_map(vendor_data)
# #folium_static(vendor_map)



# Function to calculate summary statistics
def calculate_summary_statistics(data):
    summary_stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Standard Deviation': np.std(data),
        'Total': np.sum(data)
    }
    return summary_stats

# Assuming we have a list of invoice amounts
invoice_amounts = st.sidebar('Invoice amount', min_value=0.0, format='%f')

# Calculate and display summary statistics
summary_statistics = calculate_summary_statistics(invoice_amount)
for stat, value in summary_statistics.items():
    st.metric(label=stat, value=value)

# Interactive filters in the sidebar

date_range = st.sidebar.date_input('Date Range', [])
amount_range = st.sidebar.slider('Amount Range', min_value=0, max_value=10000, value=(0, 10000))
vendor_type_filter = st.sidebar.multiselect('Vendor Type', ['Type A', 'Type B', 'Type C', 'Other'], default=['Type A'])
po_amount = st.number.input('Purchase order total', min_value=0.0, format='%f')

# Provide a stub for the filter_data_function
def filter_data_function(process_data, filters):
    # Apply filters to data
    filtered_data = process_data
    return filtered_data




@st.cache_data
def load_data():
    # Load your data here
    return data

data = load_data()

@st.cache_data
def load_data():
    # Load your data here
    return data

data = load_data()


# Assuming we have a DataFrame `historical_data` with historical invoice amounts and dates
@st.cache
def train_model(historical_data):
    # Preprocess the data for modeling
    historical_data['date_ordinal'] = historical_data['invoice_date'].apply(lambda x: x.toordinal())
    X = historical_data[['date_ordinal']]  # Features
    y = historical_data['invoice_amount']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set and calculate the error
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)

    return model, error

uploaded_file_historical = st.file_uploader('Upload historical data CSV', type='csv')
if uploaded_file_historical is not None:
    historical_data = pd.read_csv(uploaded_file_historical)
else:
    st.error('No historical data file uploaded.')
    st.stop()

# Train the model and display the error
model, error = train_model(historical_data)
st.write(f"Model trained with mean squared error: {error:.2f}")

# Use the model to make predictions
def predict_future_amounts(model, future_dates):
    future_dates['date_ordinal'] = future_dates['invoice_date'].apply(lambda x: x.toordinal())
    predictions = model.predict(future_dates[['date_ordinal']])
    return predictions


uploaded_file_future = st.file_uploader('Upload future dates CSV', type='csv')
if uploaded_file_future is not None:
    future_dates = pd.read_csv(uploaded_file_future)
else:
    st.error('No future dates file uploaded.')
    st.stop()
# Assuming `future_dates` is a DataFrame with future dates for prediction
future_amounts = predict_future_amounts(model, future_dates)
st.write("Future invoice amount predictions:", future_amounts)

import pandas as pd
import numpy as np
import plotly.express as px

# Placeholder for loading and preprocessing data
# For demonstration, creating a DataFrame with random data
np.random.seed(0)
dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()
data = pd.DataFrame({'Date': dates, 'Value': values})

# Function to update the visualization based on the user's interactions
def update_visualization(date_range):
    filtered_data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]
    fig = px.line(filtered_data, x='Date', y='Value', title='Interactive Time Series Chart')
    return fig

# Since Streamlit cannot be run in this notebook environment, we will simulate the Streamlit code
# Create the interactive elements
start_date, end_date = dates.min(), dates.max()

# Update the visualization based on the slider
fig = update_visualization((start_date, end_date))

# Display the interactive plot
fig.show()
