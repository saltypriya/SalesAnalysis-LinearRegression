#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[1]:


cd C:\Users\pysha\OneDrive\Desktop\BU\3rd semester\Probablity and Stats\dataset


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("format.csv")
df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.describe().transpose()


# In[12]:


df.count()


# In[13]:


missing_data = df.isnull()
df.dropna(axis=0, inplace=True)
count = missing_data.sum()
print(count)


# In[14]:


df.to_csv('preprocessed_data.csv', index=False)


# In[15]:


df.columns


# In[16]:


da=df.groupby("Category")
da.first()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')

df['Items_Count_Sum'] = df.groupby(df['Order Date'].dt.to_period("M"))['Order ID'].transform('count')

plt.style.use('seaborn')

df.set_index('Order Date')['Items_Count_Sum'].resample('M').mean().plot()
plt.xlabel('Month')
plt.ylabel('Mean Items Count')
plt.title('Mean Items Count per Month')
plt.grid(True)
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')

df['Items_Count_Sum'] = df.groupby(df['Order Date'].dt.to_period("M"))['Order ID'].transform('count')

plt.style.use('seaborn')

mean_values = df.set_index('Order Date')['Items_Count_Sum'].resample('M').mean()
mean_values.plot()
plt.xlabel('Month')
plt.ylabel('Mean Items Count')
plt.title('Mean Items Count per Month')
plt.grid(True)
plt.show()


# In[19]:


Sales_category=df.groupby("Category")["Sales"].sum()


Sales_category.plot(kind='bar')
plt.title('Category by Sales', fontsize = 14)
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()


# In[25]:


freq=df[['Category','Sub Category','City']].value_counts().reset_index()
freq=freq.rename(columns={0: 'frequency'})
freq


# In[26]:


pivot_table = freq.pivot_table(index='Category', columns=['Sub Category', 'City'], values='frequency')
sns.heatmap(pivot_table, cmap='Blues')
sns.set(rc={'figure.figsize':(14,8)})
plt.xticks(rotation=90)
plt.title("Frequencies distribution of Sub Category,City")


# In[45]:


ax=sns.barplot(data=freq, x='Category', y='frequency', hue='Sub Category', dodge=False)
sns.set(rc={'figure.figsize':(14,8)})
ax.legend(bbox_to_anchor=(1, 1), ncol=2)
plt.xticks(rotation=90)


# In[19]:


numrical_columns=df.select_dtypes('number')
numrical_columns


# In[20]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=numrical_columns)
plt.title('Box Plot for Numerical Columns')


# In[9]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df,x='Category',y='Profit', orient='v', fliersize=3, width=0.5)
plt.title('Box Plot for Category vs Profit')
plt.xticks(rotation=45)
plt.show()


# In[10]:


sns.boxplot(data=df,y='Profit')
plt.show()
plt.figure(figsize=(10,10))


# In[46]:


seventy_fifth=df['Profit'].quantile(0.75)
twenty_fifth=df['Profit'].quantile(0.25)
profit_IQR=seventy_fifth-twenty_fifth
print(profit_IQR)


# In[47]:


upper = seventy_fifth + (1.5 * profit_IQR)
lower = twenty_fifth - (1.5 * profit_IQR)
print(upper, lower)


# In[40]:


sns.histplot(df['Profit'])


# In[51]:


sup1=df[(df['Profit']>lower) & (df['Profit'] <upper)]


# In[52]:


sns.boxplot(data=sup1,y='Profit')
plt.figure(figsize=(4,4))
plt.show()


# In[55]:


plt.figure(figsize=(11,11))
sns.barplot(x=sup1['Sales'],y=sup1['City'])
plt.xticks(rotation=90,fontsize=13)
plt.yticks(fontsize=13)
plt.title('Grocery Sales Per City',fontweight='bold')
plt.show()


# In[58]:


sup1['Category'].value_counts()


# In[59]:


sup1['Category'].nunique()


# In[60]:


sup1['Sub Category'].value_counts()


# In[61]:


print(sup1[['Category','Sub Category']].value_counts().to_markdown())


# In[62]:


Sub_Sales_Region=pd.pivot_table(sup1,columns='Region',index=['Sub Category'],values='Sales',aggfunc=np.mean,  fill_value=0)
Sub_Sales_Region.loc[Sub_Sales_Region.mean(axis=1).sort_values(ascending=False).index]


# In[63]:


new1=Sub_Sales_Region.nlargest(5,'Central')
new1.style.bar(color='#FBEEAC')


# In[64]:


Cat_Profit=pd.pivot_table(sup1,columns=['Category'],index=['City'],values='Profit',aggfunc=np.mean)
Cat_Profit.loc[Cat_Profit.mean(axis=1).sort_values(ascending=False).index]


# In[65]:


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

styled_table = Sub_Sales_Region.style.apply(highlight_max, axis=1)

styled_table


# In[66]:


styled_table1 = Cat_Profit.style.apply(highlight_max, axis=1)

styled_table1


# In[67]:


def highlight_min(m):
    is_min = m == m.min()
    return ['background-color: red' if v else '' for v in is_min]

min_table = Cat_Profit.style.apply(highlight_min, axis=1)

min_table


# In[68]:


sup1.loc[(sup1['Customer Name']=='Amrish')&(sup1['Category']=='Bakery')&(sup1['Sales']<1000)].sort_values(by='Sales',ascending=False)


# In[69]:


sup1.loc[(sup1['Customer Name']=='Harish')&(sup1['Category']=='Food Grains')&(sup1['Sales']>2000)].sort_values(by='Sales',ascending=False)


# In[70]:


sup1.loc[(sup1['Customer Name']=='Hussain')&(sup1['Category']=='Beverages')&(sup1['Sales']>2000)].sort_values(by='Sales',ascending=False)


# In[71]:


super_group=sup1.groupby('Customer Name').agg({'Sales':'sum','Profit':'sum'})
super_group['Performance']=super_group['Sales']+super_group['Profit']
super_group = super_group.sort_values(by='Performance', ascending=False)
print("The top 5 customers are:")
print(super_group.index[:5].tolist())
print("The last 3 customers are:")
print(super_group.tail(3).index.tolist())


# In[73]:


correlation_matrix = df[['Sales', 'Profit']].corr()

print(correlation_matrix)


# In[80]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
    
X = df[['Discount', 'Profit']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(10, 6))

plt.scatter(X_test['Discount'], y_test, color='black', label='Actual Sales', alpha=0.7, edgecolors='w')
plt.scatter(X_test['Discount'], predictions, color='red', label='Predicted Sales', alpha=0.7, edgecolors='w')

plt.xlabel('Discount')
plt.ylabel('Sales')
plt.legend()
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

features = df[['Discount', 'Profit']]
target = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, hue=y_test, palette='viridis', alpha=0.7)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Linear Regression: Actual vs. Predicted Sales')
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

features = df[['Discount', 'Profit']]
target = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, palette='viridis', alpha=0.7)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Decision Tree: Actual vs. Predicted Sales')
plt.show()

