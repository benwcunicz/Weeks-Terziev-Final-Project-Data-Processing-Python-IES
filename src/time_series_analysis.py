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

#linear regression for nasa inflation-adjusted budget and the inflation rate data
x = nasa_bud_inflat_adj_df['Actual'].to_frame()
y1 = us_gdp_df['US GDP'].to_frame()

#scale GDP and inflation-adjusted buget. Use StandardScaler due to absence of outliers in inflation-adjusted budget and GDP datasets. 
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y1)

model = LinearRegression()
model.fit(x_scaled,y1_scaled)
y_pred = model.predict(x_scaled)
mse = mean_squared_error(y1_scaled, y_pred)

print(f"MSE is {mse}")
print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])


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

#matplotlib code
#plotting the raw data vs the 4 year rolling mean and raw data vs. 4 year rolling standard deviation
