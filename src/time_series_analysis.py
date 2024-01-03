import pandas as pd

#import nasa budget inflation-adjusted data from dataset and filter data by dropping unneccessary columns and rows that contain NaN data.
nasa_bud_inflat_adj_df = pd.read_excel('planetary_exploration_budget_dataset.xlsx',sheet_name='budget_history_inflation_adj')
columns_to_drop_nasa_infla_adj_budget_df = ['Request','Enacted','Discretionary','Notes']
nasa_bud_inflat_adj_df.drop(columns=columns_to_drop_nasa_infla_adj_budget_df,axis=1,inplace=True)
nasa_bud_inflat_adj_df = nasa_bud_inflat_adj_df[nasa_bud_inflat_adj_df['Actual'].notna()]
nasa_bud_inflat_adj_df = nasa_bud_inflat_adj_df[(nasa_bud_inflat_adj_df['Fiscal Year'] >= 1960)]

#import inflation rate data
inflation_rate_df = pd.read_csv('inflation_rate_data.csv')


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

#print out any outliers, do a correlation analysis against time data and also inflation rate data
outliers = check_outliers_budget(nasa_bud_inflat_adj_df)
pearson_correlation_year = nasa_bud_inflat_adj_df['Actual'].corr(nasa_bud_inflat_adj_df['Fiscal Year'])
pearson_correlation_inflation = nasa_bud_inflat_adj_df['Actual'].corr(inflation_rate_df['Inflation Rate'])
print(f"Number of outliers is {outliers}")
print(f"Pearson correlation for year is {pearson_correlation_year}")
print(f"Pearson correlation for inflation is {pearson_correlation_inflation}")


#matplotlib code for data visualization

