import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

df = pd.read_csv('coffee_sales.csv')

df.info()
df.describe()

df = df.dropna()
df = df.drop_duplicates()

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

sales_by_day = df.groupby('date').agg({'money': 'sum'}).reset_index()
sales_by_day.rename(columns = {'money':'total_sales'}, inplace = True)

sales_by_month = df.groupby('month').agg({'money': 'sum'}).reset_index()
sales_by_month.rename(columns = {'money':'total_sales'}, inplace = True)


sales_by_month['previous_month_sales'] = sales_by_month['total_sales'].shift(1)
sales_by_month = sales_by_month.dropna()

X = sales_by_month[['previous_month_sales']]
y = sales_by_month['total_sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

st.title('Coffee Sales Prediction')

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
print(RMSE)
MSE = mean_squared_error(y_test,y_pred)
print(MSE)

sales_by_month.to_csv('sales_by_month.csv',index=False)
st.write(sales_by_month)
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test.values,label = 'actual')
plt.plot(y_pred,label = 'predicted')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('monthly Sales Over Time')
plt.legend()
st.pyplot(fig)



costumer_purchase = df.groupby('card').agg({'money': 'sum', 'coffee_name': 'count'}).reset_index()
costumer_purchase.rename(columns={'money': 'total_spent', 'coffee_name': 'coffee_bought'}, inplace=True)

top_costumers = costumer_purchase.sort_values(by='total_spent', ascending=False).head(10)

# Bar chart for top customers
st.write('Top 10 Customers by Total Spent')
fig2 = plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_costumers, 
    x='card', 
    y='total_spent', 
    palette='viridis'
)

plt.xlabel('Customer Card', fontsize=12)
plt.ylabel('Total Spent', fontsize=12)
plt.title('Top 10 Customers by Total Spent', fontsize=14)
plt.tight_layout()
st.pyplot(fig2)

# Aggregate data to find the most sold coffee
coffee_sales = df.groupby('coffee_name').agg({'money': 'sum', 'coffee_name': 'count'}).rename(
    columns={'money': 'total_sales', 'coffee_name': 'quantity_sold'}).reset_index()

most_sold_coffee = coffee_sales.sort_values(by='quantity_sold', ascending=False)

st.write('Most Sold Coffee')
# Pie chart for most sold coffee
fig3 = plt.figure(figsize=(8, 8))
plt.pie(
    most_sold_coffee['quantity_sold'], 
    labels=most_sold_coffee['coffee_name'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=sns.color_palette('pastel')
)

# Add a title
plt.title('Proportion of Coffee Sales', fontsize=14)
plt.tight_layout()
st.pyplot(fig3)

st.write('Enter the previous month sales to predict the total sales for the next month.')
previous_month_sales = st.number_input('Previous Month Sales', value=0)
next_month_sales = model.predict([[previous_month_sales]])[0]
if st.button('Predict'):
    st.write(f'The predicted total sales for the next month are: {next_month_sales:.2f}')



st.write(sales_by_month)
f = plt.figure(figsize=(12, 6))
plt.plot(sales_by_month['month'], sales_by_month['total_sales'],label = 'monthly Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('monthly Sales Over Time')
plt.legend()
st.pyplot(f)

from statsmodels.tsa.arima.model import ARIMA
sales_by_month.set_index('month', inplace=True)
if not isinstance(sales_by_month.index, pd.DatetimeIndex):
    sales_by_month.index = pd.to_datetime(sales_by_month.index)

# Fit ARIMA model
arima_model = ARIMA(sales_by_month['total_sales'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()

# Forecast next month's sales
forecast_step = 1
forecast_model = arima_model_fit.forecast(steps=forecast_step)
next_month_sales = forecast_model[0]

# Determine forecast date
forecast_date = sales_by_month.index[-1] + pd.DateOffset(months=1)

# Plot actual and forecasted sales
fig4 = plt.figure(figsize=(12, 6))
plt.plot(sales_by_month.index, sales_by_month['total_sales'], label='Actual Sales', marker='o')
plt.scatter(forecast_date, next_month_sales, color='red', label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Over Time')
plt.axvline(forecast_date, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.grid()
st.pyplot(fig4)
