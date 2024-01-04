#note: a decade in the dataset is assumed to be 10 years, e.g. 1960-1970, 1970-1980, etc.
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

#import the 'cost space launches low earth orbit' dataset (abbreviated as leo)
leo_df_unfilitered = pd.read_csv('cost-space-launches-low-earth-orbit.csv')

#import NASA's annual budget per year dataset
nasa_annual_budget_unfilitered = pd.read_csv('nasa-annual-budget.csv')

#drop columns from NASA annual budget dataset
columns_to_drop_NASA_budget_df = ['Entity','Code']
#drop unneccessary columns from NASA dataset and filter by years between 1960 and 2020
nasa_budget_df = nasa_annual_budget_unfilitered[(nasa_annual_budget_unfilitered['Year']>=1960)&(nasa_annual_budget_unfilitered['Year']<=2020)].drop(columns=columns_to_drop_NASA_budget_df,axis=1)

#filter the leo dataset, by dropping the unused "Code" column
leo_df = leo_df_unfilitered.drop('Code', axis=1)
#sort values by year in ascending order
leo_df.sort_values(by='Year')

#list of launches not affiliated with nasa to drop
not_affiliated_with_nasa_list = ['Angara', 'Ariane 44', 'Ariane 5G', 'Dnepr', 'Electron', 'Epsilon', 'Falcon 1', 'GSLV', 'H-II', 'Kosmos', 'Kuaizhou', 'LVM3', 'Long March 11', 'Long March 2A', 'Long March 2C', 'Long March 2D', 'Long March 2E', 'Long March 3A', 'Long March 3B', 'Long March 4B', 'Long March 5', 'M-V', 'PSLV', 'Pegasus XL', 'Proton', 'R-36 / Cyclone', 'Rokot', 'Shavit', 'Shian Quxian', 'Shtil', 'Soyuz', 'Start', 'Strela', 'Vega', 'Zenit 2', 'Zenit 3SL']

#calculate the average cost per decade and add it to the list of average costs
def calculate_average_cost_per_decade(df, start_year, end_year, average_costs_list):
    #filter data between the start and end year and not affiliated with NASA
    cost_decade = df[(df['Year'] >= start_year) & (df['Year'] < end_year) & (~df['Entity'].isin(not_affiliated_with_nasa_list))]
    #calculate the mean and round to two decimal points
    average_cost = round(cost_decade['cost_per_kg'].mean(), 2)
    #append the result to the average cost list if it's not NaN
    if not np.isnan(average_cost):
        average_costs_list.append(average_cost)

#calculate the average NASA cost per decade and add it to the list of average budget 
def calculate_average_budget_per_decade(df,start_year,end_year,average_budget_list):
    #filter data between the start and end year
    budget_decade = df[(df['Year'] >= start_year) & (df['Year'] < end_year)]
    #calculate the mean of the budget per decade and round to two decimal points
    average_budget = round(budget_decade['Budget'].mean(),2)
    #append the result to the average budget list if it's not NaN
    if not np.isnan(average_budget):
        average_budget_list.append(average_budget)

#function for checking for outliers in NASA budget data, returns number of outliers
def check_outliers_budget(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['Budget'].quantile(0.25)
    q3 = df['Budget'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['Budget'] >= higherOutlier) | (df['Budget'] <= lowerOutlier))]
    return filterOutliers['Budget'].count()

#function for checking for outliers in LEO cost per kg data, returns number of outliers
def check_outliers_cost(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['cost_per_kg'].quantile(0.25)
    q3 = df['cost_per_kg'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['cost_per_kg'] >= higherOutlier) | (df['cost_per_kg'] <= lowerOutlier))]
    return filterOutliers['cost_per_kg'].count()

#print outliers
outliers_in_NASA_budget_count = check_outliers_budget(nasa_budget_df)
outliers_in_cost_per_kg_count = check_outliers_cost(leo_df)

print(f"The number of outliers in the cost_per_kg series is {outliers_in_cost_per_kg_count}")
print(f"The number of outliers in the NASA budget series is {outliers_in_NASA_budget_count}")

#calculate average costs per each decade of available data from the dataset 
average_costs_per_decade = []
decades = [(1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020)]
for start_year, end_year in decades:
    calculate_average_cost_per_decade(leo_df, start_year, end_year, average_costs_per_decade)

#make a series from the average costs per decade list
average_costs_per_decade_series = pd.Series(average_costs_per_decade)

average_budget_per_decade = []
for start_year,end_year in decades:
    calculate_average_budget_per_decade(nasa_budget_df,start_year,end_year,average_budget_per_decade)

#make a series from the average budget per decade list
average_budget_per_decade_series = pd.Series(average_budget_per_decade)

"""
set x and y variables for linear regression
the average annual NASA budget per decade, x, is the independent variable we are investigating to check the correlation with average cost to LEO per decade, y,
use to_frame on x and y to make x and y 2D value instead of 1D by converting it to a dataframe
"""
x = average_budget_per_decade_series.to_frame()
y = average_costs_per_decade_series.to_frame()

#use RobustScaler to scale x and y to reduce mean squared error due to large numbers in data. 
#RobustScaler is used instead of StandardScaler or MinMaxScaler due to the presence of several outliers in the datasets. 
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y)

#create a linear regression model
model = LinearRegression()
model.fit(x_scaled,y_scaled)

#make predictions for variable y
y_pred = model.predict(x_scaled)

#evaluate the mean squared error of the model
mse = mean_squared_error(y_scaled, y_pred)
print("Mean Squared Error:", mse)

#printing slope and intercept of linear regression line
print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

#calculate pearson correlation coefficient and print it
pearson_correlation = average_budget_per_decade_series.corr(average_costs_per_decade_series)
print("Pearson Correlation Coefficient:", pearson_correlation)

#matplotlib code
