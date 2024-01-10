#note: "federal spending" here means government expenditure as a percentage of GDP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler,StandardScaler

#import nasa budget inflation-adjusted data from dataset and filter data by dropping unneccessary columns and rows that contain NaN data.
nasa_bud_inflat_adj_df = pd.read_excel('planetary_exploration_budget_dataset.xlsx',sheet_name='budget_history_inflation_adj')
columns_to_drop_nasa_infla_adj_budget_df = ['Request','Enacted','Discretionary','Notes']
nasa_bud_inflat_adj_df.drop(columns=columns_to_drop_nasa_infla_adj_budget_df,axis=1,inplace=True)
nasa_bud_inflat_adj_df = nasa_bud_inflat_adj_df[nasa_bud_inflat_adj_df['Actual'].notna()]
nasa_bud_inflat_adj_df = nasa_bud_inflat_adj_df[(nasa_bud_inflat_adj_df['Fiscal Year'] >= 1960)&(nasa_bud_inflat_adj_df['Fiscal Year'] <= 2022)]

#import inflation rate data
inflation_rate_df = pd.read_csv('inflation_rate_data.csv')

#import federal spending data
federal_spending_df = pd.read_excel('federal_spending_data.xls',sheet_name='organized_spending')

#import U.S. GDP data
us_gdp_df = pd.read_excel('gdp_data.xls',sheet_name='gdp_data_us')

#function for checking for outliers in NASA inflation-adjusted budget data, returns number of outliers
def check_outliers_budget(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['Actual'].quantile(0.25)
    q3 = df['Actual'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['Actual'] >= higherOutlier) | (df['Actual'] <= lowerOutlier))]
    return filterOutliers['Actual'].count()

#function for checking for outliers in the inflation rate data, returns number of outliers
def check_outliers_inflation(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['Inflation Rate'].quantile(0.25)
    q3 = df['Inflation Rate'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['Inflation Rate'] >= higherOutlier) | (df['Inflation Rate'] <= lowerOutlier))]
    return filterOutliers['Inflation Rate'].count()

#function for checking for outliers in the federal spending data, returns number of outliers
def check_outliers_federal_spending(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['Federal Spending'].quantile(0.25)
    q3 = df['Federal Spending'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['Federal Spending'] >= higherOutlier) | (df['Federal Spending'] <= lowerOutlier))]
    return filterOutliers['Federal Spending'].count()

#function for checking for outliers in the US GDP data, returns number of outliers
def check_outliers_gdp(df):
    #calculate using outliers formula with upper and lower quartiles
    q1 = df['US GDP'].quantile(0.25)
    q3 = df['US GDP'].quantile(0.75)
    IQR = q3-q1
    higherOutlier = q3 + (1.5*IQR)
    lowerOutlier = q1 - (1.5*IQR)
    filterOutliers = df[((df['US GDP'] >= higherOutlier) | (df['US GDP'] <= lowerOutlier))]
    return filterOutliers['US GDP'].count()

#print out any outliers, do a correlation analysis against time data and also inflation rate data
outliers = check_outliers_budget(nasa_bud_inflat_adj_df)
inflation_outliers = check_outliers_inflation(inflation_rate_df)
fed_spending_outliers = check_outliers_federal_spending(federal_spending_df)
gdp_outliers = check_outliers_gdp(us_gdp_df)

#function for finding pearson correlation
def check_corr(series):
    pearson_correlation = nasa_bud_inflat_adj_df['Actual'].corr(series)
    return pearson_correlation

#find pearson correlation against the inflation-adjusted NASA budget for each varaible
pearson_correlation_year = check_corr(nasa_bud_inflat_adj_df['Fiscal Year'])
pearson_correlation_inflation = check_corr(inflation_rate_df['Inflation Rate'])
pearson_correlation_federal_spending = check_corr(federal_spending_df['Federal Spending'])
pearson_correlation_gdp = check_corr(us_gdp_df['US GDP'])

#linear regression for nasa inflation-adjusted budget and the US GDP, Federal Spending and Inflation Rate data
x = nasa_bud_inflat_adj_df['Actual'].to_frame()
y1 = us_gdp_df['US GDP'].to_frame()
y2 = federal_spending_df['Federal Spending'].to_frame()
y3 = inflation_rate_df['Inflation Rate'].to_frame()

#scale GDP and inflation-adjusted buget. Use StandardScaler due to absence of outliers in inflation-adjusted budget and GDP datasets. 
#use RobustScaler for inflation rate data due to presence of several outliers
scaler = StandardScaler()
robust_scaler = RobustScaler()

x_scaled = scaler.fit_transform(x)
x1_scaled = robust_scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y1)
y2_scaled = scaler.fit_transform(y2)
y3_scaled = robust_scaler.fit_transform(y3)

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model1.fit(x_scaled,y1_scaled)
model2.fit(x_scaled,y2_scaled)
model3.fit(x1_scaled,y3_scaled)

y1_pred = model1.predict(x_scaled)
y2_pred = model2.predict(x_scaled)
y3_pred = model3.predict(x1_scaled)

#calculate mean squared error for each model
mse = mean_squared_error(y1_scaled, y1_pred)
mse2 = mean_squared_error(y2_scaled,y2_pred)
mse3 = mean_squared_error(y3_scaled,y3_pred)

#calculate r_squared score for each model
r_squared_model_1 = model1.score(x_scaled,y1_scaled)
r_squared_model_2 = model2.score(x_scaled,y2_scaled)
r_squared_model_3 = model3.score(x_scaled,y3_scaled)

print(f"MSE is {mse}, {mse2},{mse3}")
print(f"R-squared is {r_squared_model_1},{r_squared_model_2},{r_squared_model_3}")
print(f"Slope: {model1.coef_[0][0]},{model2.coef_[0][0]},{model3.coef_[0][0]}")
print(f"Intercept: {model1.intercept_[0]}, {model2.intercept_[0]},{model3.intercept_[0]}")


#find rolling mean and standard deviation of time series data
#4 years is selected here because the length of each U.S. president's term is 4 years. 
nasa_bud_inflat_adj_df['Rolling Mean'] = nasa_bud_inflat_adj_df['Actual'].rolling(4).mean()
nasa_bud_inflat_adj_df['Rolling Standard Dev'] = nasa_bud_inflat_adj_df['Actual'].rolling(4).std()

print(f"Rolling mean is {nasa_bud_inflat_adj_df['Rolling Mean']}")
print(f"Rolling standard deviation is {nasa_bud_inflat_adj_df['Rolling Standard Dev']}")
print(f"Number of outliers is {outliers}, {inflation_outliers}, {fed_spending_outliers}, {gdp_outliers}")
print(f"Pearson correlation against time is {pearson_correlation_year}")
print(f"Pearson correlation for inflation is {pearson_correlation_inflation}")
print(f"Pearson correlation for GDP is {pearson_correlation_gdp}")
print(f"Pearson correlation for federal spending is {pearson_correlation_federal_spending}")

#Plotting the raw data vs the 4 year rolling mean and raw data vs. 4 year rolling standard deviation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

#Raw data vs. 4-year rolling mean
ax1.plot(nasa_bud_inflat_adj_df['Fiscal Year'], nasa_bud_inflat_adj_df['Actual'], label='Budget')
ax1.plot(nasa_bud_inflat_adj_df['Fiscal Year'], nasa_bud_inflat_adj_df['Rolling Mean'], label='4-Year Rolling Mean', color='red')
ax1.set_title('NASA Budget vs 4-Year Rolling Mean')
ax1.set_xlabel('Year')
ax1.set_ylabel('NASA Budget')
ax1.legend()

#Raw data vs. 4-year rolling standard deviation
ax2.plot(nasa_bud_inflat_adj_df['Fiscal Year'], nasa_bud_inflat_adj_df['Actual'], label='Budget')
ax2.plot(nasa_bud_inflat_adj_df['Fiscal Year'], nasa_bud_inflat_adj_df['Rolling Standard Dev'], label='4-Year Rolling Std. Dev.', color='orange')
ax2.set_title('NASA Budget vs 4-Year Rolling Standard Deviation')
ax2.set_xlabel('Year')
ax2.set_ylabel('NASA Budget')
ax2.legend()

plt.tight_layout()
plt.show()

#Visualizing the variables over time relative to 1960 value

years = nasa_bud_inflat_adj_df['Fiscal Year']
nasa_budget_1960_normalized = nasa_bud_inflat_adj_df['Actual'] / nasa_bud_inflat_adj_df[nasa_bud_inflat_adj_df['Fiscal Year'] == 1960]['Actual'].values[0]
inflation_rate_1960_normalized = inflation_rate_df['Inflation Rate'] / inflation_rate_df[inflation_rate_df['Year'] == 1960]['Inflation Rate'].values[0]
federal_spending_1960_normalized = federal_spending_df['Federal Spending'] / federal_spending_df[federal_spending_df['Year'] == 1960]['Federal Spending'].values[0]
gdp_1960_normalized = us_gdp_df['US GDP'] / us_gdp_df[us_gdp_df['Year'] == 1960]['US GDP'].values[0]

plt.figure(figsize=(8, 6))

# Plot each normalized data set
plt.plot(years, nasa_budget_1960_normalized, label='NASA Budget')
plt.plot(years, inflation_rate_1960_normalized, label='Inflation Rate')
plt.plot(years, federal_spending_1960_normalized, label='Federal Spending')
plt.plot(years, gdp_1960_normalized, label='GDP')

plt.title('Relevant Variables Overtime Relative to Their Value in 1960')
plt.xlabel('Year')
plt.ylabel('Relative Value (1960 = 1)')
plt.legend()
plt.tight_layout()
plt.show()

#Linear regression graphs

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# NASA Budget vs US GDP
ax1.scatter(x, y1, color='blue', label='US GDP')
ax1.plot(x, scaler.inverse_transform(y1_pred), color='red', label='Predicted US GDP')
ax1.set_title(f'NASA Budget vs US GDP (MSE: {mse:.2f}, R²: {r_squared_model_1:.2f})')
ax1.set_xlabel('NASA Budget')
ax1.set_ylabel('US GDP')
ax1.legend()

# NASA Budget vs Federal Spending
ax2.scatter(x, y2, color='blue', label='Federal Spending')
ax2.plot(x, scaler.inverse_transform(y2_pred), color='red', label='Predicted Federal Spending')
ax2.set_title(f'NASA Budget vs Federal Spending (MSE: {mse2:.2f}, R²: {r_squared_model_2:.2f})')
ax2.set_xlabel('NASA Budget')
ax2.set_ylabel('Federal Spending')
ax2.legend()

# NASA Budget vs Inflation Rate
ax3.scatter(x, y3, color='blue', label='Inflation Rate')
ax3.plot(x, robust_scaler.inverse_transform(y3_pred), color='red', label='Predicted Inflation Rate')
ax3.set_title(f'NASA Budget vs Inflation Rate (MSE: {mse3:.2f}, R²: {r_squared_model_3:.2f})')
ax3.set_xlabel('NASA Budget')
ax3.set_ylabel('Inflation Rate')
ax3.legend()

plt.tight_layout()
plt.show()
