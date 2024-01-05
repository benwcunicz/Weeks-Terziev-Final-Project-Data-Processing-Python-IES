#note: "federal spending" here means government expenditure as a percentage of GDP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

#print out any outliers, do a correlation analysis against time data and also inflation rate data
outliers = check_outliers_budget(nasa_bud_inflat_adj_df)
inflation_outliers = check_outliers_inflation(inflation_rate_df)
fed_spending_outliers = check_outliers_federal_spending(federal_spending_df)

#function for finding pearson correlation
def check_corr(series):
    pearson_correlation = nasa_bud_inflat_adj_df['Actual'].corr(series)
    return pearson_correlation

#find pearson correlation against the inflation-adjusted NASA budget for each varaible
pearson_correlation_year = check_corr(nasa_bud_inflat_adj_df['Fiscal Year'])
pearson_correlation_inflation = check_corr(inflation_rate_df['Inflation Rate'])
pearson_correlation_federal_spending = check_corr(federal_spending_df['Federal Spending'])

#linear regression for nasa inflation-adjusted budget and the inflation rate data
x = nasa_bud_inflat_adj_df['Actual'].to_frame()
y1 = inflation_rate_df['Inflation Rate'].to_frame()

model = LinearRegression()
model.fit(x,y1)
y_pred = model.predict(x)
mse = mean_squared_error(y1, y_pred)

print(f"MSE is {mse}")
print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])


#find rolling mean of time series data
#4 years is selected here because the length of each U.S. president's term is 4 years. 
nasa_bud_inflat_adj_df['Rolling Mean'] = nasa_bud_inflat_adj_df['Actual'].rolling(4).mean()

print(f"Rolling mean is {nasa_bud_inflat_adj_df['Rolling Mean']}")
print(f"Number of outliers is {outliers}, {inflation_outliers}, {fed_spending_outliers}")
print(f"Pearson correlation against time is {pearson_correlation_year}")
print(f"Pearson correlation for inflation is {pearson_correlation_inflation}")
print(f"Pearson correlation for federal spending is {pearson_correlation_federal_spending}")

#matplotlib code
#plotting the raw data vs the 4 year rolling mean
