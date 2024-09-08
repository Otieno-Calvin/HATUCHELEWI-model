# Predicting Late Deliveries : A Detailed Business Problem

![image.png](test copy_files/image.png)

### **Problem Statement:**

In the competitive e-commerce industry, timely delivery is crucial for maintaining customer satisfaction and brand reputation. Late deliveries not only frustrate customers but also lead to increased costs due to expedited shipping and potential compensation.


This dataset offers a comprehensive view of various factors influencing delivery times, such as order details, customer demographics, and shipping information.We wil be focusing on Puerto Rico for today. By predicting which shipping days, businesses can take proactive measures to ensure timely fulfillment, optimize resource allocation, and enhance overall customer experience.

#### Business Objective

Develop a predictive model that returns real shipping days to allow optimization of delivery days this will allow identification of orders likely to experience delays before they occur. This model will allow the logistics team to prioritize and intervene on high-risk shipments, adjust delivery schedules, and manage customer expectations more effectively.




**Libraries used :**
1. pandas
2. statsmodels
3. numpy
4. matplotlib
5. scikit learn
6. XGboost
7. skopt
8. category_encoders
9. shap

#### **Implementation Steps:**

1. **Data Preparation:**

- Cleaning and preprocessing the data, sorting out any missing values, outliers, or inconsistencies.
- Feature engineering to create new variables that could enhance model performance, such as calculating the number of orders shipped and number of orders made on a particular day as traffic.


1. **Exploratory Data Analysis (EDA):**

- Conduct an in-depth analysis to understand the distribution of delivery times and the correlation between variables.
- Identify key predictors of late deliveries.


3. **Model Development:**

- Train and test various machine learning models (e.g., logistic regression, random forests, gradient boosting) to predict the likelihood of late deliveries.
- Evaluate model performance using metrics such as precision, recall, and the F1 score.

#### Import libraries


```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as  plt
import seaborn as sns
```

#### Load dataset


```python
# Import pandas to work with the DataFrame
import pandas as pd

# Read the dataset from the CSV file. 
# 'Unnamed: 0' column is set as the index .
df = pd.read_csv('Dataset/dataset.csv', index_col='Unnamed: 0')

# Display the first 20 rows of the DataFrame to check the data content and structure.
df.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Days_for_shipping_(real)</th>
      <th>Days_for_shipment_(scheduled)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Delivery_Status</th>
      <th>Late_delivery_risk</th>
      <th>Category_Id</th>
      <th>Category_Name</th>
      <th>Customer_City</th>
      <th>...</th>
      <th>Order_Zipcode</th>
      <th>Product_Card_Id</th>
      <th>Product_Category_Id</th>
      <th>Product_Description</th>
      <th>Product_Image</th>
      <th>Product_Name</th>
      <th>Product_Price</th>
      <th>Product_Status</th>
      <th>shipping_date_(DateOrders)</th>
      <th>Shipping_Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>91.25000</td>
      <td>314.64001</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>2/3/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>-249.09000</td>
      <td>311.35999</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/18/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>-247.78000</td>
      <td>309.72000</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>San Jose</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/17/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>22.86000</td>
      <td>304.81000</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/16/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>4</td>
      <td>134.21001</td>
      <td>298.25000</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRANSFER</td>
      <td>6</td>
      <td>4</td>
      <td>18.58000</td>
      <td>294.98001</td>
      <td>Shipping canceled</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Tonawanda</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/19/2018</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>1</td>
      <td>95.18000</td>
      <td>288.42001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRANSFER</td>
      <td>2</td>
      <td>1</td>
      <td>68.43000</td>
      <td>285.14001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Miami</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CASH</td>
      <td>3</td>
      <td>2</td>
      <td>133.72000</td>
      <td>278.59000</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/16/2018</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CASH</td>
      <td>2</td>
      <td>1</td>
      <td>132.14999</td>
      <td>275.31000</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>San Ramon</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TRANSFER</td>
      <td>6</td>
      <td>2</td>
      <td>130.58000</td>
      <td>272.03000</td>
      <td>Shipping canceled</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/19/2018</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>2</td>
      <td>45.69000</td>
      <td>268.76001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Freeport</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/18/2018</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TRANSFER</td>
      <td>4</td>
      <td>2</td>
      <td>21.76000</td>
      <td>262.20001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Salinas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/17/2018</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>13</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>1</td>
      <td>24.58000</td>
      <td>245.81000</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRANSFER</td>
      <td>2</td>
      <td>1</td>
      <td>16.39000</td>
      <td>327.75000</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Peabody</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>1</td>
      <td>-259.57999</td>
      <td>324.47000</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PAYMENT</td>
      <td>5</td>
      <td>2</td>
      <td>-246.36000</td>
      <td>321.20001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Canovanas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/18/2018</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CASH</td>
      <td>2</td>
      <td>1</td>
      <td>23.84000</td>
      <td>317.92001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Paramount</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>1</td>
      <td>102.26000</td>
      <td>314.64001</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018</td>
      <td>First Class</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PAYMENT</td>
      <td>0</td>
      <td>0</td>
      <td>87.18000</td>
      <td>311.35999</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Mount Prospect</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/13/2018</td>
      <td>Same Day</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 53 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 123595 entries, 0 to 123594
    Data columns (total 53 columns):
     #   Column                         Non-Null Count   Dtype  
    ---  ------                         --------------   -----  
     0   Type                           123595 non-null  object 
     1   Days_for_shipping_(real)       123595 non-null  int64  
     2   Days_for_shipment_(scheduled)  123595 non-null  int64  
     3   Benefit_per_order              123595 non-null  float64
     4   Sales_per_customer             123595 non-null  float64
     5   Delivery_Status                123595 non-null  object 
     6   Late_delivery_risk             123595 non-null  int64  
     7   Category_Id                    123595 non-null  int64  
     8   Category_Name                  123595 non-null  object 
     9   Customer_City                  123595 non-null  object 
     10  Customer_Country               123595 non-null  object 
     11  Customer_Email                 123595 non-null  object 
     12  Customer_Fname                 123595 non-null  object 
     13  Customer_Id                    123595 non-null  int64  
     14  Customer_Lname                 123589 non-null  object 
     15  Customer_Password              123595 non-null  object 
     16  Customer_Segment               123595 non-null  object 
     17  Customer_State                 123595 non-null  object 
     18  Customer_Street                123595 non-null  object 
     19  Customer_Zipcode               123592 non-null  float64
     20  Department_Id                  123595 non-null  int64  
     21  Department_Name                123595 non-null  object 
     22  Latitude                       123595 non-null  float64
     23  Longitude                      123595 non-null  float64
     24  Market                         123595 non-null  object 
     25  Order_City                     123595 non-null  object 
     26  Order_Country                  123595 non-null  object 
     27  Order_Customer_Id              123595 non-null  int64  
     28  order_date_(DateOrders)        123595 non-null  object 
     29  Order_Id                       123595 non-null  int64  
     30  Order_Item_Cardprod_Id         123595 non-null  int64  
     31  Order_Item_Discount            123595 non-null  float64
     32  Order_Item_Discount_Rate       123595 non-null  float64
     33  Order_Item_Id                  123595 non-null  int64  
     34  Order_Item_Product_Price       123595 non-null  int64  
     35  Order_Item_Profit_Ratio        123595 non-null  float64
     36  Order_Item_Quantity            123595 non-null  int64  
     37  Sales                          123595 non-null  float64
     38  Order_Item_Total               123595 non-null  float64
     39  Order_Profit_Per_Order         123595 non-null  float64
     40  Order_Region                   123595 non-null  object 
     41  Order_State                    123595 non-null  object 
     42  Order_Status                   123595 non-null  object 
     43  Order_Zipcode                  16521 non-null   float64
     44  Product_Card_Id                123595 non-null  int64  
     45  Product_Category_Id            123595 non-null  int64  
     46  Product_Description            0 non-null       float64
     47  Product_Image                  123595 non-null  object 
     48  Product_Name                   123595 non-null  object 
     49  Product_Price                  123595 non-null  float64
     50  Product_Status                 123595 non-null  int64  
     51  shipping_date_(DateOrders)     123595 non-null  object 
     52  Shipping_Mode                  123595 non-null  object 
    dtypes: float64(14), int64(15), object(24)
    memory usage: 50.9+ MB
    


```python
# This will generate descriptive statistics for all numeric columns in the dataframe
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Days_for_shipping_(real)</th>
      <th>Days_for_shipment_(scheduled)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Late_delivery_risk</th>
      <th>Category_Id</th>
      <th>Customer_Id</th>
      <th>Customer_Zipcode</th>
      <th>Department_Id</th>
      <th>Latitude</th>
      <th>...</th>
      <th>Order_Item_Quantity</th>
      <th>Sales</th>
      <th>Order_Item_Total</th>
      <th>Order_Profit_Per_Order</th>
      <th>Order_Zipcode</th>
      <th>Product_Card_Id</th>
      <th>Product_Category_Id</th>
      <th>Product_Description</th>
      <th>Product_Price</th>
      <th>Product_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123592.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>...</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>16521.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>0.0</td>
      <td>123595.000000</td>
      <td>123595.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.479032</td>
      <td>2.864978</td>
      <td>21.090146</td>
      <td>175.096074</td>
      <td>0.564287</td>
      <td>31.824572</td>
      <td>6706.230438</td>
      <td>39913.130882</td>
      <td>5.415413</td>
      <td>30.982568</td>
      <td>...</td>
      <td>2.121243</td>
      <td>194.866545</td>
      <td>175.096074</td>
      <td>21.090146</td>
      <td>55426.449912</td>
      <td>691.377928</td>
      <td>31.824572</td>
      <td>NaN</td>
      <td>132.954527</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.624559</td>
      <td>1.381694</td>
      <td>100.832643</td>
      <td>116.884299</td>
      <td>0.495852</td>
      <td>15.862559</td>
      <td>4203.202333</td>
      <td>37583.720377</td>
      <td>1.636375</td>
      <td>9.513307</td>
      <td>...</td>
      <td>1.474254</td>
      <td>128.800632</td>
      <td>116.884299</td>
      <td>100.832643</td>
      <td>31991.430193</td>
      <td>340.186158</td>
      <td>15.862559</td>
      <td>NaN</td>
      <td>131.371339</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-4274.979980</td>
      <td>7.490000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>603.000000</td>
      <td>2.000000</td>
      <td>-33.937550</td>
      <td>...</td>
      <td>1.000000</td>
      <td>9.990000</td>
      <td>7.490000</td>
      <td>-4274.980000</td>
      <td>1040.000000</td>
      <td>19.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>9.990000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>6.720000</td>
      <td>103.990000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>3244.000000</td>
      <td>725.000000</td>
      <td>4.000000</td>
      <td>18.282110</td>
      <td>...</td>
      <td>1.000000</td>
      <td>119.980000</td>
      <td>103.990000</td>
      <td>6.720000</td>
      <td>23320.000000</td>
      <td>403.000000</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>30.670000</td>
      <td>161.970000</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>6453.000000</td>
      <td>30066.000000</td>
      <td>5.000000</td>
      <td>33.891650</td>
      <td>...</td>
      <td>1.000000</td>
      <td>199.920000</td>
      <td>161.970000</td>
      <td>30.670000</td>
      <td>59405.000000</td>
      <td>627.000000</td>
      <td>29.000000</td>
      <td>NaN</td>
      <td>84.400000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>62.390000</td>
      <td>227.960010</td>
      <td>1.000000</td>
      <td>46.000000</td>
      <td>9819.000000</td>
      <td>80126.000000</td>
      <td>7.000000</td>
      <td>39.837920</td>
      <td>...</td>
      <td>3.000000</td>
      <td>250.000000</td>
      <td>227.960010</td>
      <td>62.390000</td>
      <td>90008.000000</td>
      <td>1014.000000</td>
      <td>46.000000</td>
      <td>NaN</td>
      <td>199.990000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>864.000000</td>
      <td>1919.989990</td>
      <td>1.000000</td>
      <td>76.000000</td>
      <td>20757.000000</td>
      <td>99205.000000</td>
      <td>12.000000</td>
      <td>48.781930</td>
      <td>...</td>
      <td>5.000000</td>
      <td>1999.989990</td>
      <td>1919.989990</td>
      <td>864.000000</td>
      <td>99301.000000</td>
      <td>1363.000000</td>
      <td>76.000000</td>
      <td>NaN</td>
      <td>1999.990000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>




```python
# dropping duplicate records

df.drop_duplicates(inplace=True)
```


```python
# Show columns
df.columns
```




    Index(['Type', 'Days_for_shipping_(real)', 'Days_for_shipment_(scheduled)',
           'Benefit_per_order', 'Sales_per_customer', 'Delivery_Status',
           'Late_delivery_risk', 'Category_Id', 'Category_Name', 'Customer_City',
           'Customer_Country', 'Customer_Email', 'Customer_Fname', 'Customer_Id',
           'Customer_Lname', 'Customer_Password', 'Customer_Segment',
           'Customer_State', 'Customer_Street', 'Customer_Zipcode',
           'Department_Id', 'Department_Name', 'Latitude', 'Longitude', 'Market',
           'Order_City', 'Order_Country', 'Order_Customer_Id',
           'order_date_(DateOrders)', 'Order_Id', 'Order_Item_Cardprod_Id',
           'Order_Item_Discount', 'Order_Item_Discount_Rate', 'Order_Item_Id',
           'Order_Item_Product_Price', 'Order_Item_Profit_Ratio',
           'Order_Item_Quantity', 'Sales', 'Order_Item_Total',
           'Order_Profit_Per_Order', 'Order_Region', 'Order_State', 'Order_Status',
           'Order_Zipcode', 'Product_Card_Id', 'Product_Category_Id',
           'Product_Description', 'Product_Image', 'Product_Name', 'Product_Price',
           'Product_Status', 'shipping_date_(DateOrders)', 'Shipping_Mode'],
          dtype='object')



Below are the columns used in my model.

| Column Name                  | Description                                                                 | Assumed Impact on Deliveries                                                                 |
|------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Type**                      | Type of payment.                                                            | Certain types of payment may require longer and various verification methods.      |
| **Benefit_per_order**        | Profit margin per order.                                                     | Higher benefit orders might be prioritized in shipping to maximize revenue.          |
| **Sales_per_customer**       | Total sales value per customer.                                              | High-value customers may receive better shipping as a loyalty strategy.           |
| **Category_Id**              | Identifier for product categories.                                           | Some categories may be more prone to delays due to size, weight, or regulation.      |
| **Customer_City**            | City of the customer.                                                        | Geographic location can impact shipping time due to distance or logistics network.    |
| **Customer_Country**         | Country of the customer.                                                     | International shipments may face delays due to customs or cross-border regulations.   |
| **Customer_State**           | State of the customer.                                                       | Regional differences in infrastructure can affect delivery times.                    |
| **Department_Id**            | Identifier for the department handling the product.                          | Different departments may have varying efficiencies, affecting order processing times.|
| **Market**                   | Market segment for the product.                                              | Certain markets may have faster shipping due to better infrastructure or priority.    |
| **Order_City**               | City where the order was placed.                                             | Similar to `Customer_City`, affects logistics and delivery speed.                    |
| **Order_Country**            | Country where the order was placed.                                          | Affects delivery speed and logistics, especially for international orders.           |
| **Order_Item_Product_Price** | Price of individual order items.                                             | Higher-priced items may receive priority shipping to ensure customer satisfaction.    |
| **Order_Item_Profit_Ratio**  | Profit ratio of individual order items.                                       | Items with higher profit margins might be prioritized in the shipping process.        |
| **Order_Item_Quantity**      | Quantity of items in the order.                                              | Bulk orders may require different shipping methods, potentially causing delays.      |
| **Order_Item_Total**         | Total value of the order.                                                    | High-value orders may receive priority in shipping.                                  |
| **Order_Region**             | Region where the order was placed.                                           | Regional logistics networks can impact delivery times.                               |
| **Order_State**              | State where the order was placed.                                            | Similar to `Customer_State`, affects delivery speed due to regional logistics.       |
| **Order_Status**             | Status of the order (e.g., shipped, delivered, pending).                     | Delays in status change could indicate potential delivery issues.                    |
| **Product_Price**            | Price of the product.                                                        | Expensive products may be shipped faster to maintain customer satisfaction.          |
| **order_date_(DateOrders)**  | Date the order was placed.                                                   | Orders placed on weekends or holidays may experience delays.                         |
| **order_date**               | Same as `order_date_(DateOrders)`, but formatted differently.                | As above, impacts shipping time.                                                     |
| **shipping_date**            | Date the order was shipped.                                                  | The difference between order and shipping date indicates processing time.            |
| **order_day_of_week**        | Day of the week the order was placed.                                        | Orders placed on certain days may be processed slower (e.g., weekends).              |
| **order_month**              | Month the order was placed.                                                  | Peak seasons (e.g., holidays) may lead to longer processing times.                   |
| **shipping_day_of_week**     | Day of the week the order was shipped.                                       | Similar to `order_day_of_week`, affects delivery speed.                              |
| **shipping_month**           | Month the order was shipped.                                                 | Shipping during peak seasons may result in delays.                                   |
| **Orders_Made_That_Day**     | Total number of orders made on that day.                                     | High order volumes can strain logistics, leading to delays.                          |
| **Orders_Shipped_That_Day**  | Total number of orders shipped on that day.                                  | High shipping volumes could lead to delays if logistics capacity is exceeded.        |



dropping irrelevant columns 


```python
import pandas as pd

# Assuming your DataFrame is named 'df'
columns_to_drop = [
    'Delivery_Status',
    'Customer_Fname',
    'Customer_Lname',
    'Customer_Password',
    'Customer_Segment',
    'Customer_Street',
    'Customer_Zipcode',
    'Customer_Id',
    'Department_Name',
    'Latitude',
    'Late_delivery_risk',
    'Days_for_shipment_(scheduled)',
    'Longitude',
    'Order_Customer_Id',
    'Order_Id',
    'Order_Item_Cardprod_Id',
    'Order_Item_Discount',
    'Order_Item_Discount_Rate',
    'Order_Item_Id',
    'Sales',
    'Customer_Email',
    'Order_Profit_Per_Order',
    'Category_Name',
    'Product_Card_Id',
    'Product_Description',
    'Product_Image',
    'Product_Name',
    'Product_Status',
    'shipping_date_(DateOrders)',
    'Shipping_Mode',
    'Order_Zipcode',
    'Product_Category_Id'  ,
]

# Drop the specified columns
df_selected = df.drop(columns=columns_to_drop)


```


```python
df_selected.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 123595 entries, 0 to 123594
    Data columns (total 31 columns):
     #   Column                    Non-Null Count   Dtype         
    ---  ------                    --------------   -----         
     0   Type                      123595 non-null  object        
     1   Days_for_shipping_(real)  123595 non-null  int64         
     2   Benefit_per_order         123595 non-null  float64       
     3   Sales_per_customer        123595 non-null  float64       
     4   Category_Id               123595 non-null  int64         
     5   Customer_City             123595 non-null  object        
     6   Customer_Country          123595 non-null  object        
     7   Customer_State            123595 non-null  object        
     8   Department_Id             123595 non-null  int64         
     9   Market                    123595 non-null  object        
     10  Order_City                123595 non-null  object        
     11  Order_Country             123595 non-null  object        
     12  order_date_(DateOrders)   123595 non-null  object        
     13  Order_Item_Product_Price  123595 non-null  int64         
     14  Order_Item_Profit_Ratio   123595 non-null  float64       
     15  Order_Item_Quantity       123595 non-null  int64         
     16  Order_Item_Total          123595 non-null  float64       
     17  Order_Region              123595 non-null  object        
     18  Order_State               123595 non-null  object        
     19  Order_Status              123595 non-null  object        
     20  Product_Price             123595 non-null  float64       
     21  order_date                123595 non-null  datetime64[ns]
     22  shipping_date             123595 non-null  datetime64[ns]
     23  order_day_of_week         123595 non-null  object        
     24  order_month               123595 non-null  int32         
     25  order_year                123595 non-null  int32         
     26  shipping_day_of_week      123595 non-null  object        
     27  shipping_month            123595 non-null  int32         
     28  shipping_year             123595 non-null  int32         
     29  Orders_Made_That_Day      123595 non-null  int64         
     30  Orders_Shipped_That_Day   123595 non-null  int64         
    dtypes: datetime64[ns](2), float64(5), int32(4), int64(7), object(13)
    memory usage: 27.3+ MB
    


```python
# Calculate the correlation of numeric columns in 'df_selected' with 'Days_for_shipping_(real)'
df_selected.corr(numeric_only=True)['Days_for_shipping_(real)']

```




    Days_for_shipping_(real)    1.000000
    Benefit_per_order          -0.004754
    Sales_per_customer          0.002315
    Category_Id                 0.007584
    Department_Id               0.005844
    Order_Item_Product_Price    0.004163
    Order_Item_Profit_Ratio    -0.004533
    Order_Item_Quantity        -0.001775
    Order_Item_Total            0.002315
    Product_Price               0.004162
    order_month                -0.006952
    order_year                 -0.001642
    shipping_month             -0.003936
    shipping_year               0.002819
    Orders_Made_That_Day       -0.002537
    Orders_Shipped_That_Day    -0.040277
    Name: Days_for_shipping_(real), dtype: float64




```python
# Find the distibution of the df (mean)
df_selected.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Days_for_shipping_(real)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Category_Id</th>
      <th>Department_Id</th>
      <th>Order_Item_Product_Price</th>
      <th>Order_Item_Profit_Ratio</th>
      <th>Order_Item_Quantity</th>
      <th>Order_Item_Total</th>
      <th>Product_Price</th>
      <th>order_date</th>
      <th>shipping_date</th>
      <th>order_month</th>
      <th>order_year</th>
      <th>shipping_month</th>
      <th>shipping_year</th>
      <th>Orders_Made_That_Day</th>
      <th>Orders_Shipped_That_Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595</td>
      <td>123595</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
      <td>123595.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.479032</td>
      <td>21.090146</td>
      <td>175.096074</td>
      <td>31.824572</td>
      <td>5.415413</td>
      <td>132.966811</td>
      <td>0.121654</td>
      <td>2.121243</td>
      <td>175.096074</td>
      <td>132.954527</td>
      <td>2016-06-22 08:54:38.346211584</td>
      <td>2016-06-25 20:24:26.738945792</td>
      <td>6.271176</td>
      <td>2015.995825</td>
      <td>6.289987</td>
      <td>2016.003965</td>
      <td>115.892399</td>
      <td>116.621748</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-4274.979980</td>
      <td>7.490000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>-2.750000</td>
      <td>1.000000</td>
      <td>7.490000</td>
      <td>9.990000</td>
      <td>2015-01-01 00:00:00</td>
      <td>2015-01-03 00:00:00</td>
      <td>1.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>2015.000000</td>
      <td>36.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>6.720000</td>
      <td>103.990000</td>
      <td>18.000000</td>
      <td>4.000000</td>
      <td>50.000000</td>
      <td>0.080000</td>
      <td>1.000000</td>
      <td>103.990000</td>
      <td>50.000000</td>
      <td>2015-09-18 00:00:00</td>
      <td>2015-09-22 00:00:00</td>
      <td>3.000000</td>
      <td>2015.000000</td>
      <td>3.000000</td>
      <td>2015.000000</td>
      <td>106.000000</td>
      <td>104.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>30.670000</td>
      <td>161.970000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>84.000000</td>
      <td>0.270000</td>
      <td>1.000000</td>
      <td>161.970000</td>
      <td>84.400000</td>
      <td>2016-07-03 00:00:00</td>
      <td>2016-07-07 00:00:00</td>
      <td>6.000000</td>
      <td>2016.000000</td>
      <td>6.000000</td>
      <td>2016.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>62.390000</td>
      <td>227.960010</td>
      <td>46.000000</td>
      <td>7.000000</td>
      <td>200.000000</td>
      <td>0.360000</td>
      <td>3.000000</td>
      <td>227.960010</td>
      <td>199.990000</td>
      <td>2017-03-22 00:00:00</td>
      <td>2017-03-25 00:00:00</td>
      <td>9.000000</td>
      <td>2017.000000</td>
      <td>9.000000</td>
      <td>2017.000000</td>
      <td>130.000000</td>
      <td>132.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>864.000000</td>
      <td>1919.989990</td>
      <td>76.000000</td>
      <td>12.000000</td>
      <td>2000.000000</td>
      <td>0.500000</td>
      <td>5.000000</td>
      <td>1919.989990</td>
      <td>1999.990000</td>
      <td>2018-01-31 00:00:00</td>
      <td>2018-02-06 00:00:00</td>
      <td>12.000000</td>
      <td>2018.000000</td>
      <td>12.000000</td>
      <td>2018.000000</td>
      <td>175.000000</td>
      <td>209.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.624559</td>
      <td>100.832643</td>
      <td>116.884299</td>
      <td>15.862559</td>
      <td>1.636375</td>
      <td>131.372402</td>
      <td>0.465460</td>
      <td>1.474254</td>
      <td>116.884299</td>
      <td>131.371339</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.392444</td>
      <td>0.842548</td>
      <td>3.387520</td>
      <td>0.844937</td>
      <td>22.180111</td>
      <td>24.882811</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

# Set number of rows and columns for the grid
n_rows = 8
n_cols = 4

# Create a grid of subplots (2x2 in this case)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))

# Adjust spacing between subplots (play around with these values)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.25, hspace=0.35)

# Flatten the grid array for easy access to subplots
axs = axs.flatten()

# Loop through each selected column in the DataFrame
for i, column in enumerate(df_selected.columns):
    # Plot each column against actual shipping days
    axs[i].scatter(df_selected[column], df_selected['Days_for_shipping_(real)'])
    
    # Label x-axis as the current column
    axs[i].set_xlabel(column)
    
    # Label y-axis as 'Actual shipping days'
    axs[i].set_ylabel('Actual shipping days')
    
    # Set title for each subplot
    axs[i].set_title(f'Scatter plot of {column} vs Actual shipping days')

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

# Display the grid of scatter plots
plt.show()

```


    
![png](test%20copy_files/test%20copy_17_0.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size for the plots
plt.figure(figsize=(9, 9))

# Plot delivery times over time
sns.lineplot(x='order_date', y='Days_for_shipping_(real)', data=df_selected)
plt.title('Delivery Times Over Time')  # Title 
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()  # Display the plot

# Plot delivery times by month
sns.boxplot(x='order_month', y='Days_for_shipping_(real)', data=df_selected)
plt.title('Delivery Times by Month')  # Title of the box plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()  # Display the plot

# Plot delivery times by day of the week
sns.boxplot(x='order_day_of_week', y='Days_for_shipping_(real)', data=df_selected)
plt.title('Delivery Times by Day of Week')  # Title of the box plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()  # Display the plot

```


    
![png](test%20copy_files/test%20copy_18_0.png)
    



    
![png](test%20copy_files/test%20copy_18_1.png)
    



    
![png](test%20copy_files/test%20copy_18_2.png)
    



To improve the accuracy of the analysis, the Isolation Forest algorithm is used to detect and remove outliers from the dataset. This step helps ensure that only consistent data is included, allowing for more reliable insights into Benefit_per_order and Sales_per_customer.


```python
from sklearn.ensemble import IsolationForest

# Initialize Isolation Forest with 1% contamination rate and a fixed random state for reproducibility
iso_forest = IsolationForest(contamination=0.01, random_state=42)

# Fit the model on 'Benefit_per_order' and 'Sales_per_customer' columns to detect outliers
outliers = iso_forest.fit_predict(df_selected[['Benefit_per_order', 'Sales_per_customer']])

# Keep only the rows where the prediction is not marked as an outlier (-1 represents an outlier)
df_selected = df_selected[outliers != -1]

```

ensuring consistency across the dataset ,we improve model performance by selecting numerical columns which are then standardized using a scaler. This process adjusts the values of Benefit_per_order, Sales_per_customer, Order_Item_Product_Price, and Product_Price to have a mean of zero and a standard deviation of one, helping to normalize variations and enhance analysis.



```python
from sklearn.preprocessing import StandardScaler

# Define the columns that need to be scaled
columns_to_scale = ['Benefit_per_order', 'Sales_per_customer', 'Order_Item_Product_Price', 'Product_Price']

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler and transform the selected columns
df_selected[columns_to_scale] = scaler.fit_transform(df_selected[columns_to_scale])


```


```python
# viewing the first to entries in the dataframe
df_selected.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Days_for_shipping_(real)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Category_Id</th>
      <th>Customer_City</th>
      <th>Customer_Country</th>
      <th>Customer_State</th>
      <th>Department_Id</th>
      <th>Market</th>
      <th>...</th>
      <th>order_date</th>
      <th>shipping_date</th>
      <th>order_day_of_week</th>
      <th>order_month</th>
      <th>order_year</th>
      <th>shipping_day_of_week</th>
      <th>shipping_month</th>
      <th>shipping_year</th>
      <th>Orders_Made_That_Day</th>
      <th>Orders_Shipped_That_Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>0.867367</td>
      <td>1.487424</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-31</td>
      <td>2018-02-03</td>
      <td>Wednesday</td>
      <td>1</td>
      <td>2018</td>
      <td>Saturday</td>
      <td>2</td>
      <td>2018</td>
      <td>51</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>-3.591162</td>
      <td>1.453578</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-18</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Thursday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>-3.574001</td>
      <td>1.436655</td>
      <td>73</td>
      <td>San Jose</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-17</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Wednesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>-0.028557</td>
      <td>1.385989</td>
      <td>73</td>
      <td>Los Angeles</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-16</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>1.430153</td>
      <td>1.318298</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRANSFER</td>
      <td>6</td>
      <td>-0.084626</td>
      <td>1.284555</td>
      <td>73</td>
      <td>Tonawanda</td>
      <td>EE. UU.</td>
      <td>NY</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-19</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Friday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>61</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>0.918851</td>
      <td>1.216863</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TRANSFER</td>
      <td>2</td>
      <td>0.568420</td>
      <td>1.183018</td>
      <td>73</td>
      <td>Miami</td>
      <td>EE. UU.</td>
      <td>FL</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CASH</td>
      <td>3</td>
      <td>1.423733</td>
      <td>1.115429</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-16</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>67</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CASH</td>
      <td>2</td>
      <td>1.403166</td>
      <td>1.081583</td>
      <td>73</td>
      <td>San Ramon</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TRANSFER</td>
      <td>6</td>
      <td>1.382599</td>
      <td>1.047737</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-19</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Friday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>0.270521</td>
      <td>1.013995</td>
      <td>73</td>
      <td>Freeport</td>
      <td>EE. UU.</td>
      <td>NY</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-18</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Thursday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>59</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TRANSFER</td>
      <td>4</td>
      <td>-0.042967</td>
      <td>0.946303</td>
      <td>73</td>
      <td>Salinas</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-17</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Wednesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>-0.006025</td>
      <td>0.777177</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRANSFER</td>
      <td>2</td>
      <td>-0.113316</td>
      <td>1.622704</td>
      <td>73</td>
      <td>Peabody</td>
      <td>EE. UU.</td>
      <td>MA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>-3.728583</td>
      <td>1.588858</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PAYMENT</td>
      <td>5</td>
      <td>-3.555399</td>
      <td>1.555115</td>
      <td>73</td>
      <td>Canovanas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-18</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Thursday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>59</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CASH</td>
      <td>2</td>
      <td>-0.015719</td>
      <td>1.521270</td>
      <td>73</td>
      <td>Paramount</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DEBIT</td>
      <td>2</td>
      <td>1.011600</td>
      <td>1.487424</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-15</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PAYMENT</td>
      <td>0</td>
      <td>0.814049</td>
      <td>1.453578</td>
      <td>73</td>
      <td>Mount Prospect</td>
      <td>EE. UU.</td>
      <td>IL</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>2018-01-13</td>
      <td>2018-01-13</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>Saturday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 31 columns</p>
</div>



##### Exploratory Data Analysis (EDA)


```python
import pandas as pd
import matplotlib.pyplot as plt

# Group the data by order_date and calculate the average shipping time
grouped_data = df.groupby('order_date').agg({'Days_for_shipping_(real)': 'mean', 'Orders_Made_That_Day': 'sum'}).reset_index()

# Create a scatter plot
plt.scatter(grouped_data['Orders_Made_That_Day'], grouped_data['Days_for_shipping_(real)'])
plt.xlabel('Orders Made That Day')
plt.ylabel('Average Shipping Time (Days)')
plt.title('Relationship Between Order Volume and Shipping Time')
plt.show()
```


    
![png](test%20copy_files/test%20copy_25_0.png)
    



```python
# Group the data by Market and calculate the average shipping time
grouped_data = df.groupby('Market').agg({'Days_for_shipping_(real)': 'mean'}).reset_index()

# Sort the data to get the key markets
top_3_markets_highest = grouped_data.sort_values(by='Days_for_shipping_(real)', ascending=False).head(3)
top_3_markets_lowest = grouped_data.sort_values(by='Days_for_shipping_(real)', ascending=True).head(3)

print("Top 3 markets with highest shipping time:")
print(top_3_markets_highest)

print("\nTop 3 markets with lowest shipping time:")
print(top_3_markets_lowest)

# Create a box plot for visual representation
sns.boxplot(x='Market', y='Days_for_shipping_(real)', data=df)
plt.xlabel('Market')
plt.ylabel('Shipping Time (Days)')
plt.title('Distribution of Shipping Time by Market')
plt.show()

```

    Top 3 markets with highest shipping time:
             Market  Days_for_shipping_(real)
    2         LATAM                  3.487798
    0        Africa                  3.483495
    3  Pacific Asia                  3.477958
    
    Top 3 markets with lowest shipping time:
             Market  Days_for_shipping_(real)
    4          USCA                  3.462314
    1        Europe                  3.477909
    3  Pacific Asia                  3.477958
    


    
![png](test%20copy_files/test%20copy_26_1.png)
    



```python
# Group by Category_Id to get total sales or quantity sold
top_20_categories = df.groupby('Category_Id').agg({'Sales_per_customer': 'sum'}).reset_index()

# Sort by sales to get the top 20 categories
top_20_categories = top_20_categories.sort_values(by='Sales_per_customer', ascending=False).head(20)

# Filter the original DataFrame to include only these top 20 categories
df_top_20 = df[df['Category_Id'].isin(top_20_categories['Category_Id'])]

# Group by Category_Id and calculate average shipping time for the top 20 categories
grouped_by_category_top_20 = df_top_20.groupby('Category_Id').agg({'Days_for_shipping_(real)': 'mean'}).reset_index()

# Plot the boxplot for the top 20 categories
plt.figure(figsize=(10,8))
sns.boxplot(x='Category_Id', y='Days_for_shipping_(real)', data=df_top_20)
plt.title('Shipping Time by Top 20 Highest Selling Category IDs')
plt.show()

```


    
![png](test%20copy_files/test%20copy_27_0.png)
    



```python

```


```python

# 2. Payment Type
grouped_by_type = df.groupby('Type').agg({'Days_for_shipping_(real)': 'mean'}).reset_index()
sns.boxplot(x='Type', y='Days_for_shipping_(real)', data=df)
plt.title('Shipping Time by Payment Type')
plt.show()

# 3. Benefit per Order
plt.scatter(df['Benefit_per_order'], df['Days_for_shipping_(real)'])
plt.xlabel('Benefit per Order')
plt.ylabel('Shipping Time (Days)')
plt.title('Relationship Between Benefit per Order and Shipping Time')
plt.show()
```


    
![png](test%20copy_files/test%20copy_29_0.png)
    



    
![png](test%20copy_files/test%20copy_29_1.png)
    



```python


# Group by order day of week and calculate average shipping time, then sort in descending order
grouped_by_order_day = df.groupby('order_day_of_week').agg({'Days_for_shipping_(real)': 'mean'}).reset_index()
grouped_by_order_day = grouped_by_order_day.sort_values('Days_for_shipping_(real)', ascending=False)

# Group by shipping day of week and calculate average shipping time, then sort in descending order
grouped_by_shipping_day = df.groupby('shipping_day_of_week').agg({'Days_for_shipping_(real)': 'mean'}).reset_index()
grouped_by_shipping_day = grouped_by_shipping_day.sort_values('Days_for_shipping_(real)', ascending=False)

# Create bar plots
sns.barplot(x='order_day_of_week', y='Days_for_shipping_(real)', data=grouped_by_order_day)
plt.title('Shipping Time by Order Day of Week (Largest to Smallest)')
plt.show()

sns.barplot(x='shipping_day_of_week', y='Days_for_shipping_(real)', data=grouped_by_shipping_day)
plt.title('Shipping Time by Shipping Day of Week (Largest to Smallest)')
plt.show()

```


    
![png](test%20copy_files/test%20copy_30_0.png)
    



    
![png](test%20copy_files/test%20copy_30_1.png)
    


#### **Feature engineering**


##### Notes on Feature Engineering


1. **Price Binning**: Divided `Order_Item_Product_Price` into categories like Low, Medium, and High. This makes it easier for the model to handle and recognize patterns in different price ranges.

2. **Log Transformation**: Used a log transformation on `Order_Item_Profit_Ratio` to manage extreme values and reduce skewness. This helps the model work with more balanced data.

3. **Quantity Binning**: I grouped `Order_Item_Quantity` into common ranges like 1, 2, 3, etc. This simplifies the data and helps the model focus on the most common quantities.

4. **Total Binning**: categorized `Order_Item_Total` into bins such as Low, Medium, and High. This makes it easier to spot trends and simplifies the data for the model.

5. **Interaction Term**: Added a new feature combining `Log_Order_Item_Profit_Ratio` and `Order_Item_Product_Price`. This helps capture how profit and price interact, providing extra insights for better predictions.

The changes make the data easier to work with and help the model identify key patterns and trends.


```python
# 1. Create bins for Order_Item_Product_Price based on its distribution
# Since the data has a wide range of values, we group it into discrete bins to simplify it for the model
df_selected['Price_Binned'] = pd.cut(df_selected['Order_Item_Product_Price'], 
                                     bins=[-np.inf, -0.7, 0, 0.7, np.inf], 
                                     labels=['Low', 'Below_Average', 'Above_Average', 'High'])

# 2. Apply a log transformation to Order_Item_Profit_Ratio to handle extreme outliers and skewness
# A log transformation is appropriate given the wide range of values
df_selected['Log_Order_Item_Profit_Ratio'] = np.log1p(df_selected['Order_Item_Profit_Ratio'])

# 3. Simplify Order_Item_Quantity by grouping frequent values into a few categories
# This helps to reduce the large number of unique values, focusing on the most common quantities
df_selected['Quantity_Binned'] = pd.cut(df_selected['Order_Item_Quantity'], 
                                        bins=[0, 1, 2, 3, 4, 5, np.inf], 
                                        labels=['1', '2', '3', '4', '5', 'More than 5'])

# 4. For Order_Item_Total, group values into bins based on their observed frequency
# This reduces complexity while preserving the core information about total value
df_selected['Total_Binned'] = pd.cut(df_selected['Order_Item_Total'], 
                                     bins=[-np.inf, 100, 150, 200, np.inf], 
                                     labels=['Low', 'Medium', 'High', 'Very_High'])

# 5. Interaction between Log_Order_Item_Profit_Ratio and Price_Binned
# To capture how profit ratios differ across price ranges, create an interaction term
df_selected['Profit_Ratio_x_Price_Binned'] = df_selected['Log_Order_Item_Profit_Ratio'] * df_selected['Order_Item_Product_Price']

# Drop original columns that have been transformed to avoid multicollinearity
df_selected.drop(columns=['Order_Item_Product_Price', 'Order_Item_Profit_Ratio', 
                          'Order_Item_Quantity', 'Order_Item_Total'], inplace=True)

# Display the updated dataframe 
df_selected.head()
```

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\arraylike.py:399: RuntimeWarning: invalid value encountered in log1p
      result = getattr(ufunc, method)(*inputs, **kwargs)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Days_for_shipping_(real)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Category_Id</th>
      <th>Customer_City</th>
      <th>Customer_Country</th>
      <th>Customer_State</th>
      <th>Department_Id</th>
      <th>Market</th>
      <th>...</th>
      <th>shipping_day_of_week</th>
      <th>shipping_month</th>
      <th>shipping_year</th>
      <th>Orders_Made_That_Day</th>
      <th>Orders_Shipped_That_Day</th>
      <th>Price_Binned</th>
      <th>Log_Order_Item_Profit_Ratio</th>
      <th>Quantity_Binned</th>
      <th>Total_Binned</th>
      <th>Profit_Ratio_x_Price_Binned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>0.867367</td>
      <td>1.487424</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>Saturday</td>
      <td>2</td>
      <td>2018</td>
      <td>51</td>
      <td>24</td>
      <td>High</td>
      <td>0.254642</td>
      <td>1</td>
      <td>Very_High</td>
      <td>0.471771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>-3.591162</td>
      <td>1.453578</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>Thursday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>59</td>
      <td>High</td>
      <td>-1.609438</td>
      <td>1</td>
      <td>Very_High</td>
      <td>-2.981777</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>-3.574001</td>
      <td>1.436655</td>
      <td>73</td>
      <td>San Jose</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>Wednesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
      <td>High</td>
      <td>-1.609438</td>
      <td>1</td>
      <td>Very_High</td>
      <td>-2.981777</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>-0.028557</td>
      <td>1.385989</td>
      <td>73</td>
      <td>Los Angeles</td>
      <td>EE. UU.</td>
      <td>CA</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>67</td>
      <td>High</td>
      <td>0.076961</td>
      <td>1</td>
      <td>Very_High</td>
      <td>0.142584</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>1.430153</td>
      <td>1.318298</td>
      <td>73</td>
      <td>Caguas</td>
      <td>Puerto Rico</td>
      <td>PR</td>
      <td>2</td>
      <td>Pacific Asia</td>
      <td>...</td>
      <td>Monday</td>
      <td>1</td>
      <td>2018</td>
      <td>68</td>
      <td>62</td>
      <td>High</td>
      <td>0.371564</td>
      <td>1</td>
      <td>Very_High</td>
      <td>0.688389</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



The categorical columns are converted into one-hot encoded format to make them usable for analysis. This process creates new binary columns for each category in the selected columns, such as Type, Order_Region, order_day_of_week, shipping_day_of_week, and Market. Dropping the first category prevents issues with multicollinearity in the model.


```python
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import pandas as pd

def one_hot_encode_data(df):
    # Specify categorical columns to apply one-hot encoding. 
    # These columns have limited unique values, so this encoding works well.
    columns_to_encode = ['Type', 'Order_Region', 'Market', 'Order_Status']
    
    # Apply one-hot encoding to the specified columns.
    # This creates new binary columns for each category, while dropping the first category to avoid multicollinearity.
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, dtype=int, drop_first=True)

    # Apply label encoding to 'order_day_of_week' and 'shipping_day_of_week'.
    # Each unique value in these columns is assigned a numerical label.
    label_encoder = LabelEncoder()
    label_encoder_pbin = LabelEncoder()
    label_encoder_order = LabelEncoder()
    label_encoder_shipping = LabelEncoder()
    
    # Convert the binned columns to numeric values
    df_encoded['Quantity_Binned'] = label_encoder.fit_transform(df['Quantity_Binned'])
    df_encoded['Total_Binned'] = label_encoder.fit_transform(df['Total_Binned'])
    df_encoded['Order_Day_of_Week_Ordinal'] = label_encoder_order.fit_transform(df['order_day_of_week'])
    df_encoded['Shipping_Day_of_Week_Ordinal'] = label_encoder_shipping.fit_transform(df['shipping_day_of_week'])
    df_encoded['Price_Binned'] = label_encoder_pbin.fit_transform(df['Price_Binned'])

    # Perform frequency encoding for 'Order_City'.
    # Each city is represented by its frequency of occurrence in the dataset.
    order_city_freq = df['Order_City'].value_counts(normalize=True)
    df_encoded['Order_City_Frequency_Encoded'] = df['Order_City'].map(order_city_freq)


    # Apply target encoding for 'Customer_City' and 'Order_Country'.
    # Target encoding assigns values based on the mean of the target variable ('Days_for_shipping_(real)').
    # This helps capture the relationship between the categorical variable and the target variable.
    target_encoder = ce.TargetEncoder(cols=['Customer_City', 'Order_Country'])
    df_encoded[['Customer_City_Target_Encoded', 'Order_Country_Target_Encoded']] = target_encoder.fit_transform(
        df[['Customer_City', 'Order_Country']], df['Days_for_shipping_(real)'])

    return df_encoded

# Call the function to encode the selected dataframe
df_selected = one_hot_encode_data(df_selected)

```


```python
# Filter the dataframe for rows where Customer_Country is Puerto Rico
df_puerto_rico = df_selected[df_selected['Customer_Country'] == 'Puerto Rico']


df_puerto_rico.drop(columns=['Customer_State','Customer_Country'],inplace = True)

# Display the first few rows to confirm the filter
df_puerto_rico.head(20)
```

    C:\Users\SEVEN\AppData\Local\Temp\ipykernel_7608\3902272840.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_puerto_rico.drop(columns=['Customer_State','Customer_Country'],inplace = True)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Days_for_shipping_(real)</th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Category_Id</th>
      <th>Customer_City</th>
      <th>Department_Id</th>
      <th>Order_City</th>
      <th>Order_Country</th>
      <th>order_date_(DateOrders)</th>
      <th>Order_State</th>
      <th>...</th>
      <th>Order_Status_PAYMENT_REVIEW</th>
      <th>Order_Status_PENDING</th>
      <th>Order_Status_PENDING_PAYMENT</th>
      <th>Order_Status_PROCESSING</th>
      <th>Order_Status_SUSPECTED_FRAUD</th>
      <th>Order_Day_of_Week_Ordinal</th>
      <th>Shipping_Day_of_Week_Ordinal</th>
      <th>Order_City_Frequency_Encoded</th>
      <th>Customer_City_Target_Encoded</th>
      <th>Order_Country_Target_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.867367</td>
      <td>1.487424</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Bekasi</td>
      <td>Indonesia</td>
      <td>1/31/2018</td>
      <td>Java Occidental</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>0.000768</td>
      <td>3.563726</td>
      <td>3.440439</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>-3.591162</td>
      <td>1.453578</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Bikaner</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Rajastan</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0.000065</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1.430153</td>
      <td>1.318298</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Townsville</td>
      <td>Australia</td>
      <td>1/13/2018</td>
      <td>Queensland</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.000499</td>
      <td>3.563726</td>
      <td>3.469747</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>0.918851</td>
      <td>1.216863</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Guangzhou</td>
      <td>China</td>
      <td>1/13/2018</td>
      <td>Guangdong</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.001054</td>
      <td>3.563726</td>
      <td>3.471352</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>1.423733</td>
      <td>1.115429</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Guangzhou</td>
      <td>China</td>
      <td>1/13/2018</td>
      <td>Guangdong</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0.001054</td>
      <td>3.563726</td>
      <td>3.471352</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>1.382599</td>
      <td>1.047737</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Tokio</td>
      <td>Japon</td>
      <td>1/13/2018</td>
      <td>Tokio</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.000850</td>
      <td>3.563726</td>
      <td>3.508403</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>-0.006025</td>
      <td>0.777177</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Sangli</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Maharashtra</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.000180</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>-3.728583</td>
      <td>1.588858</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Sangli</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Maharashtra</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.000180</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>-3.555399</td>
      <td>1.555115</td>
      <td>73</td>
      <td>Canovanas</td>
      <td>2</td>
      <td>Seul</td>
      <td>Corea del Sur</td>
      <td>1/13/2018</td>
      <td>Seul</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0.001430</td>
      <td>3.491140</td>
      <td>3.589532</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>1.011600</td>
      <td>1.487424</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Jabalpur</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Madhya Pradesh</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.000123</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>0.750120</td>
      <td>1.385989</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Jabalpur</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Madhya Pradesh</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0.000123</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>-0.096154</td>
      <td>1.284555</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Geelong</td>
      <td>Australia</td>
      <td>1/13/2018</td>
      <td>Victoria</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0.001406</td>
      <td>3.563726</td>
      <td>3.469747</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6</td>
      <td>1.390328</td>
      <td>1.183018</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Mandurah</td>
      <td>Australia</td>
      <td>1/13/2018</td>
      <td>Australia Occidental</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.000572</td>
      <td>3.563726</td>
      <td>3.469747</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4</td>
      <td>0.753919</td>
      <td>1.081583</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Guilin</td>
      <td>China</td>
      <td>1/13/2018</td>
      <td>Guangxi</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0.000074</td>
      <td>3.563726</td>
      <td>3.471352</td>
    </tr>
    <tr>
      <th>30</th>
      <td>6</td>
      <td>0.633790</td>
      <td>0.946303</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Guilin</td>
      <td>China</td>
      <td>1/13/2018</td>
      <td>Guangxi</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.000074</td>
      <td>3.563726</td>
      <td>3.471352</td>
    </tr>
    <tr>
      <th>32</th>
      <td>4</td>
      <td>-0.006025</td>
      <td>1.622704</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Delhi</td>
      <td>India</td>
      <td>1/13/2018</td>
      <td>Delhi</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0.000948</td>
      <td>3.563726</td>
      <td>3.479062</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>1.671066</td>
      <td>1.521270</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Singapur</td>
      <td>Singapur</td>
      <td>1/13/2018</td>
      <td>Singapur</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.002321</td>
      <td>3.563726</td>
      <td>3.355634</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0.130480</td>
      <td>1.436655</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Wollongong</td>
      <td>Australia</td>
      <td>1/12/2018</td>
      <td>Nueva Gales del Sur</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.001831</td>
      <td>3.563726</td>
      <td>3.469747</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>1.449541</td>
      <td>1.284555</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Singapur</td>
      <td>Singapur</td>
      <td>1/12/2018</td>
      <td>Singapur</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.002321</td>
      <td>3.563726</td>
      <td>3.355634</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>1.240854</td>
      <td>1.183018</td>
      <td>73</td>
      <td>Caguas</td>
      <td>2</td>
      <td>Medan</td>
      <td>Indonesia</td>
      <td>1/12/2018</td>
      <td>Sumatra Septentrional</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.001667</td>
      <td>3.563726</td>
      <td>3.440439</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 68 columns</p>
</div>




```python
# Loop through each column in the dataframe
for x in df_puerto_rico.columns:
        # Print the count of each unique value in the current column        
        print (df_puerto_rico[x].value_counts())
```

    Days_for_shipping_(real)
    2    11797
    6     6360
    3     6338
    4     6308
    5     6105
    0      846
    1      763
    Name: count, dtype: int64
    Benefit_per_order
    -0.328028    243
     0.615188     70
     0.506456     50
     0.803831     50
     0.285062     49
                ... 
    -0.365757      1
    -0.225977      1
    -0.226894      1
    -0.245366      1
     0.368118      1
    Name: count, Length: 11813, dtype: int64
    Sales_per_customer
    -0.211575    371
     0.097990    351
     0.201179    347
     0.036077    346
    -0.005198    343
                ... 
    -1.269671      1
    -1.388234      1
    -1.400514      1
    -1.408769      1
    -0.054935      1
    Name: count, Length: 2219, dtype: int64
    Category_Id
    48    5936
    17    4884
    46    4632
    18    4477
    24    4292
    43    2610
    9     2392
    29    2214
    45    1700
    36     416
    35     410
    37     382
    40     349
    26     249
    7      244
    13     219
    74     213
    41     206
    33     187
    32     182
    75     153
    63     137
    73     137
    3      126
    62     118
    76     118
    67     110
    30     101
    72     101
    65      95
    66      94
    44      93
    12      84
    68      84
    59      77
    31      74
    71      73
    38      70
    69      68
    6       64
    5       61
    11      55
    61      49
    60      41
    70      41
    2       30
    34      24
    10      20
    16      12
    4        9
    64       4
    Name: count, dtype: int64
    Customer_City
    Caguas           37167
    San Juan           177
    Bayamon            145
    Humacao            110
    Yauco               92
    Manati              86
    Rio Grande          85
    Mayaguez            79
    Arecibo             71
    Carolina            70
    Guaynabo            58
    Vega Baja           57
    Trujillo Alto       55
    Juana Diaz          52
    Guayama             41
    San Sebastian       39
    Aguadilla           33
    Toa Alta            25
    Canovanas           24
    Toa Baja            21
    Ponce               15
    Cayey               15
    Name: count, dtype: int64
    Department_Id
    7     15184
    4      9792
    5      6755
    3      2782
    6      2485
    2       671
    9       385
    10      217
    11      101
    8        77
    12       68
    Name: count, dtype: int64
    Order_City
    Santo Domingo    515
    New York City    454
    Tegucigalpa      386
    Los Angeles      380
    Managua          326
                    ... 
    Rohtak             1
    Revere             1
    Belleville         1
    Broken Arrow       1
    Montgeron          1
    Name: count, Length: 2832, dtype: int64
    Order_Country
    Estados Unidos       5131
    Francia              3064
    Mexico               2972
    Alemania             2130
    Brasil               1768
                         ... 
    Barein                  1
    Guinea Ecuatorial       1
    Yibuti                  1
    Kuwait                  1
    Sri Lanka               1
    Name: count, Length: 148, dtype: int64
    order_date_(DateOrders)
    9/29/2017     70
    9/21/2017     70
    8/21/2017     67
    8/1/2017      67
    7/20/2017     66
                  ..
    1/6/2018       6
    10/26/2017     4
    12/23/2017     3
    10/30/2017     1
    10/27/2017     1
    Name: count, Length: 1125, dtype: int64
    Order_State
    Inglaterra                     1449
    California                     1040
    Isla de Francia                1020
    Renania del Norte-Westfalia     744
    San Salvador                    681
                                   ... 
    Moscu                             1
    Tokat                             1
    Yibuti                            1
    Almaty                            1
    Dublin del Sur                    1
    Name: count, Length: 954, dtype: int64
    Product_Price
     0.665786     5940
    -0.632430     4938
    -0.725252     4632
     0.016678     4529
    -0.725067     4292
                  ... 
     0.563876       10
    -0.169152        8
     12.720735       4
     4.374972        4
     8.084159        1
    Name: count, Length: 74, dtype: int64
    order_date
    2017-09-29    70
    2017-09-21    70
    2017-08-21    67
    2017-08-01    67
    2017-07-20    66
                  ..
    2018-01-06     6
    2017-10-26     4
    2017-12-23     3
    2017-10-30     1
    2017-10-27     1
    Name: count, Length: 1125, dtype: int64
    shipping_date
    2017-09-27    78
    2015-02-28    73
    2017-07-25    71
    2017-06-05    69
    2017-08-30    69
                  ..
    2017-10-29     5
    2018-02-06     4
    2018-02-05     2
    2017-10-31     2
    2017-11-01     2
    Name: count, Length: 1131, dtype: int64
    order_day_of_week
    Friday       5639
    Sunday       5567
    Thursday     5525
    Tuesday      5494
    Monday       5460
    Saturday     5445
    Wednesday    5387
    Name: count, dtype: int64
    order_month
    1     3831
    8     3640
    9     3541
    5     3493
    7     3430
    3     3316
    6     3308
    4     3118
    2     3011
    10    2650
    12    2648
    11    2531
    Name: count, dtype: int64
    order_year
    2015    12902
    2016    12630
    2017    12496
    2018      489
    Name: count, dtype: int64
    shipping_day_of_week
    Wednesday    5595
    Tuesday      5538
    Thursday     5518
    Sunday       5492
    Saturday     5478
    Monday       5466
    Friday       5430
    Name: count, dtype: int64
    shipping_month
    1     3667
    8     3645
    5     3523
    9     3507
    7     3397
    3     3346
    6     3339
    2     3098
    4     3052
    10    2856
    12    2637
    11    2450
    Name: count, dtype: int64
    shipping_year
    2015    12790
    2016    12595
    2017    12590
    2018      542
    Name: count, dtype: int64
    Orders_Made_That_Day
    124    1268
    119    1051
    121     979
    115     961
    111     938
           ... 
    39       19
    59       15
    40       11
    41       10
    37        8
    Name: count, Length: 120, dtype: int64
    Orders_Shipped_That_Day
    118    1001
    125     936
    112     927
    120     888
    114     880
           ... 
    69       10
    26        9
    24        6
    8         4
    16        2
    Name: count, Length: 138, dtype: int64
    Price_Binned
    3    14366
    0    10700
    1     8016
    2     5435
    Name: count, dtype: int64
    Log_Order_Item_Profit_Ratio
     0.392042    1943
     0.300105    1738
     0.292670    1421
     0.231112    1401
     0.385262    1381
                 ... 
    -1.966113       7
    -0.616186       5
    -2.302585       5
    -0.673345       3
    -2.659260       2
    Name: count, Length: 125, dtype: int64
    Quantity_Binned
    0    21216
    4     4609
    3     4314
    2     4231
    1     4147
    Name: count, dtype: int64
    Total_Binned
    3    10497
    0    10019
    1     9221
    2     8780
    Name: count, dtype: int64
    Profit_Ratio_x_Price_Binned
    -0.284298    448
    -0.217627    418
    -0.212236    343
    -0.279381    320
    -0.167596    306
                ... 
     0.031126      1
    -0.010168      1
     0.228031      1
    -0.059545      1
    -3.516262      1
    Name: count, Length: 2859, dtype: int64
    Type_DEBIT
    0    20370
    1    18147
    Name: count, dtype: int64
    Type_PAYMENT
    0    29625
    1     8892
    Name: count, dtype: int64
    Type_TRANSFER
    0    29999
    1     8518
    Name: count, dtype: int64
    Order_Region_Caribbean
    0    36612
    1     1905
    Name: count, dtype: int64
    Order_Region_Central Africa
    0    38164
    1      353
    Name: count, dtype: int64
    Order_Region_Central America
    0    32283
    1     6234
    Name: count, dtype: int64
    Order_Region_Central Asia
    0    38394
    1      123
    Name: count, dtype: int64
    Order_Region_East Africa
    0    38099
    1      418
    Name: count, dtype: int64
    Order_Region_East of USA
    0    37087
    1     1430
    Name: count, dtype: int64
    Order_Region_Eastern Asia
    0    37114
    1     1403
    Name: count, dtype: int64
    Order_Region_Eastern Europe
    0    37657
    1      860
    Name: count, dtype: int64
    Order_Region_North Africa
    0    37848
    1      669
    Name: count, dtype: int64
    Order_Region_Northern Europe
    0    36411
    1     2106
    Name: count, dtype: int64
    Order_Region_Oceania
    0    36697
    1     1820
    Name: count, dtype: int64
    Order_Region_South America
    0    35193
    1     3324
    Name: count, dtype: int64
    Order_Region_South Asia
    0    36890
    1     1627
    Name: count, dtype: int64
    Order_Region_South of  USA
    0    37678
    1      839
    Name: count, dtype: int64
    Order_Region_Southeast Asia
    0    36776
    1     1741
    Name: count, dtype: int64
    Order_Region_Southern Africa
    0    38274
    1      243
    Name: count, dtype: int64
    Order_Region_Southern Europe
    0    36453
    1     2064
    Name: count, dtype: int64
    Order_Region_US Center
    0    37306
    1     1211
    Name: count, dtype: int64
    Order_Region_West Africa
    0    37736
    1      781
    Name: count, dtype: int64
    Order_Region_West Asia
    0    37189
    1     1328
    Name: count, dtype: int64
    Order_Region_West of USA
    0    36866
    1     1651
    Name: count, dtype: int64
    Order_Region_Western Europe
    0    32349
    1     6168
    Name: count, dtype: int64
    Market_Europe
    0    27319
    1    11198
    Name: count, dtype: int64
    Market_LATAM
    0    27054
    1    11463
    Name: count, dtype: int64
    Market_Pacific Asia
    0    30475
    1     8042
    Name: count, dtype: int64
    Market_USCA
    0    33167
    1     5350
    Name: count, dtype: int64
    Order_Status_CLOSED
    0    35557
    1     2960
    Name: count, dtype: int64
    Order_Status_COMPLETE
    0    22959
    1    15558
    Name: count, dtype: int64
    Order_Status_ON_HOLD
    0    35928
    1     2589
    Name: count, dtype: int64
    Order_Status_PAYMENT_REVIEW
    0    38147
    1      370
    Name: count, dtype: int64
    Order_Status_PENDING
    0    35053
    1     3464
    Name: count, dtype: int64
    Order_Status_PENDING_PAYMENT
    0    29995
    1     8522
    Name: count, dtype: int64
    Order_Status_PROCESSING
    0    34818
    1     3699
    Name: count, dtype: int64
    Order_Status_SUSPECTED_FRAUD
    0    37820
    1      697
    Name: count, dtype: int64
    Order_Day_of_Week_Ordinal
    0    5639
    3    5567
    4    5525
    5    5494
    1    5460
    2    5445
    6    5387
    Name: count, dtype: int64
    Shipping_Day_of_Week_Ordinal
    6    5595
    5    5538
    4    5518
    3    5492
    2    5478
    1    5466
    0    5430
    Name: count, dtype: int64
    Order_City_Frequency_Encoded
    0.012815    515
    0.000098    509
    0.000163    469
    0.011834    454
    0.000270    425
               ... 
    0.000793     35
    0.000785     32
    0.000809     29
    0.001242     29
    0.001144     21
    Name: count, Length: 246, dtype: int64
    Customer_City_Target_Encoded
    3.563726    37167
    3.350283      177
    3.834481      145
    3.427279      110
    3.901857       92
    3.256116       86
    3.447105       85
    3.670359       79
    3.954839       71
    3.755274       70
    3.651294       58
    3.080013       57
    4.161185       55
    3.628478       52
    3.312892       41
    3.597500       39
    3.721497       33
    3.529012       25
    3.491140       24
    3.452014       21
    3.423334       15
    3.196809       15
    Name: count, dtype: int64
    Order_Country_Target_Encoded
    3.466679    5131
    3.497723    3064
    3.509363    2972
    3.487221    2130
    3.554651    1768
                ... 
    3.545848       2
    2.968057       2
    3.415740       1
    3.789970       1
    3.620327       1
    Name: count, Length: 146, dtype: int64
    


```python
df_puerto_rico.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 38517 entries, 0 to 123594
    Data columns (total 68 columns):
     #   Column                        Non-Null Count  Dtype         
    ---  ------                        --------------  -----         
     0   Days_for_shipping_(real)      38517 non-null  int64         
     1   Benefit_per_order             38517 non-null  float64       
     2   Sales_per_customer            38517 non-null  float64       
     3   Category_Id                   38517 non-null  int64         
     4   Customer_City                 38517 non-null  object        
     5   Department_Id                 38517 non-null  int64         
     6   Order_City                    38517 non-null  object        
     7   Order_Country                 38517 non-null  object        
     8   order_date_(DateOrders)       38517 non-null  object        
     9   Order_State                   38517 non-null  object        
     10  Product_Price                 38517 non-null  float64       
     11  order_date                    38517 non-null  datetime64[ns]
     12  shipping_date                 38517 non-null  datetime64[ns]
     13  order_day_of_week             38517 non-null  object        
     14  order_month                   38517 non-null  int32         
     15  order_year                    38517 non-null  int32         
     16  shipping_day_of_week          38517 non-null  object        
     17  shipping_month                38517 non-null  int32         
     18  shipping_year                 38517 non-null  int32         
     19  Orders_Made_That_Day          38517 non-null  int64         
     20  Orders_Shipped_That_Day       38517 non-null  int64         
     21  Price_Binned                  38517 non-null  int32         
     22  Log_Order_Item_Profit_Ratio   37418 non-null  float64       
     23  Quantity_Binned               38517 non-null  int32         
     24  Total_Binned                  38517 non-null  int32         
     25  Profit_Ratio_x_Price_Binned   37418 non-null  float64       
     26  Type_DEBIT                    38517 non-null  int32         
     27  Type_PAYMENT                  38517 non-null  int32         
     28  Type_TRANSFER                 38517 non-null  int32         
     29  Order_Region_Caribbean        38517 non-null  int32         
     30  Order_Region_Central Africa   38517 non-null  int32         
     31  Order_Region_Central America  38517 non-null  int32         
     32  Order_Region_Central Asia     38517 non-null  int32         
     33  Order_Region_East Africa      38517 non-null  int32         
     34  Order_Region_East of USA      38517 non-null  int32         
     35  Order_Region_Eastern Asia     38517 non-null  int32         
     36  Order_Region_Eastern Europe   38517 non-null  int32         
     37  Order_Region_North Africa     38517 non-null  int32         
     38  Order_Region_Northern Europe  38517 non-null  int32         
     39  Order_Region_Oceania          38517 non-null  int32         
     40  Order_Region_South America    38517 non-null  int32         
     41  Order_Region_South Asia       38517 non-null  int32         
     42  Order_Region_South of  USA    38517 non-null  int32         
     43  Order_Region_Southeast Asia   38517 non-null  int32         
     44  Order_Region_Southern Africa  38517 non-null  int32         
     45  Order_Region_Southern Europe  38517 non-null  int32         
     46  Order_Region_US Center        38517 non-null  int32         
     47  Order_Region_West Africa      38517 non-null  int32         
     48  Order_Region_West Asia        38517 non-null  int32         
     49  Order_Region_West of USA      38517 non-null  int32         
     50  Order_Region_Western Europe   38517 non-null  int32         
     51  Market_Europe                 38517 non-null  int32         
     52  Market_LATAM                  38517 non-null  int32         
     53  Market_Pacific Asia           38517 non-null  int32         
     54  Market_USCA                   38517 non-null  int32         
     55  Order_Status_CLOSED           38517 non-null  int32         
     56  Order_Status_COMPLETE         38517 non-null  int32         
     57  Order_Status_ON_HOLD          38517 non-null  int32         
     58  Order_Status_PAYMENT_REVIEW   38517 non-null  int32         
     59  Order_Status_PENDING          38517 non-null  int32         
     60  Order_Status_PENDING_PAYMENT  38517 non-null  int32         
     61  Order_Status_PROCESSING       38517 non-null  int32         
     62  Order_Status_SUSPECTED_FRAUD  38517 non-null  int32         
     63  Order_Day_of_Week_Ordinal     38517 non-null  int32         
     64  Shipping_Day_of_Week_Ordinal  38517 non-null  int32         
     65  Order_City_Frequency_Encoded  38517 non-null  float64       
     66  Customer_City_Target_Encoded  38517 non-null  float64       
     67  Order_Country_Target_Encoded  38517 non-null  float64       
    dtypes: datetime64[ns](2), float64(8), int32(46), int64(5), object(7)
    memory usage: 13.5+ MB
    


```python
df_puerto_rico.dropna(inplace=True)
```

    C:\Users\SEVEN\AppData\Local\Temp\ipykernel_7608\427217003.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_puerto_rico.dropna(inplace=True)
    


```python
numeric_only_df = df_puerto_rico.select_dtypes(include= [float ,int])
```

#### **Model selection , Methodology and HyperParameter tuning**

for our first model only the numeric columns are used as features, excluding the target column Days_for_shipping_(real), which is the variable being predicted. The data is split into training and testing sets, with 20% reserved for testing to evaluate model performance. Adding a constant ensures the model includes an intercept term. An Ordinary Least Squares (OLS) regression model is then created and trained using the training data.

##### 1. Statsmodels Linear Model (OLS)


```python
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Define feature columns by dropping the target column and selecting only numeric columns
X = numeric_only_df.drop(columns=['Days_for_shipping_(real)']).select_dtypes(include=[float, int]) # Feature columns

# Define the target column
y = numeric_only_df['Days_for_shipping_(real)'] # Target column

# Split the data into training and testing sets
# - test_size=0.2: 20% of the data will be used for testing
# - random_state=42: Ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the model (intercept term)
X_train = sm.add_constant(X_train)

# Create and fit the Ordinary Least Squares (OLS) regression model
lm = sm.OLS(y_train, X_train).fit()

# Print the summary of the fitted model
lm.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Days_for_shipping_(real)</td> <th>  R-squared:         </th> <td>   0.040</td> 
</tr>
<tr>
  <th>Model:</th>                       <td>OLS</td>           <th>  Adj. R-squared:    </th> <td>   0.038</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>Least Squares</td>      <th>  F-statistic:       </th> <td>   24.25</td> 
</tr>
<tr>
  <th>Date:</th>                 <td>Sun, 08 Sep 2024</td>     <th>  Prob (F-statistic):</th> <td>2.37e-220</td>
</tr>
<tr>
  <th>Time:</th>                     <td>16:26:49</td>         <th>  Log-Likelihood:    </th> <td> -55882.</td> 
</tr>
<tr>
  <th>No. Observations:</th>          <td> 29934</td>          <th>  AIC:               </th> <td>1.119e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>              <td> 29882</td>          <th>  BIC:               </th> <td>1.123e+05</td>
</tr>
<tr>
  <th>Df Model:</th>                  <td>    51</td>          <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>          <td>nonrobust</td>        <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                        <td>   30.7010</td> <td>   16.373</td> <td>    1.875</td> <td> 0.061</td> <td>   -1.391</td> <td>   62.793</td>
</tr>
<tr>
  <th>Benefit_per_order</th>            <td>    0.0273</td> <td>    0.027</td> <td>    1.026</td> <td> 0.305</td> <td>   -0.025</td> <td>    0.079</td>
</tr>
<tr>
  <th>Sales_per_customer</th>           <td>    0.0134</td> <td>    0.028</td> <td>    0.479</td> <td> 0.632</td> <td>   -0.042</td> <td>    0.068</td>
</tr>
<tr>
  <th>Category_Id</th>                  <td>   -0.0025</td> <td>    0.001</td> <td>   -1.752</td> <td> 0.080</td> <td>   -0.005</td> <td>    0.000</td>
</tr>
<tr>
  <th>Department_Id</th>                <td>    0.0017</td> <td>    0.012</td> <td>    0.136</td> <td> 0.892</td> <td>   -0.022</td> <td>    0.026</td>
</tr>
<tr>
  <th>Product_Price</th>                <td>    0.0009</td> <td>    0.032</td> <td>    0.027</td> <td> 0.978</td> <td>   -0.061</td> <td>    0.063</td>
</tr>
<tr>
  <th>order_month</th>                  <td>   -0.7648</td> <td>    0.029</td> <td>  -26.019</td> <td> 0.000</td> <td>   -0.822</td> <td>   -0.707</td>
</tr>
<tr>
  <th>order_year</th>                   <td>   -9.3106</td> <td>    0.340</td> <td>  -27.360</td> <td> 0.000</td> <td>   -9.978</td> <td>   -8.644</td>
</tr>
<tr>
  <th>shipping_month</th>               <td>    0.7572</td> <td>    0.029</td> <td>   25.784</td> <td> 0.000</td> <td>    0.700</td> <td>    0.815</td>
</tr>
<tr>
  <th>shipping_year</th>                <td>    9.2876</td> <td>    0.340</td> <td>   27.308</td> <td> 0.000</td> <td>    8.621</td> <td>    9.954</td>
</tr>
<tr>
  <th>Orders_Made_That_Day</th>         <td>    0.0007</td> <td>    0.001</td> <td>    1.146</td> <td> 0.252</td> <td>   -0.000</td> <td>    0.002</td>
</tr>
<tr>
  <th>Orders_Shipped_That_Day</th>      <td>   -0.0042</td> <td>    0.000</td> <td>   -8.665</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.003</td>
</tr>
<tr>
  <th>Price_Binned</th>                 <td>    0.0263</td> <td>    0.010</td> <td>    2.724</td> <td> 0.006</td> <td>    0.007</td> <td>    0.045</td>
</tr>
<tr>
  <th>Log_Order_Item_Profit_Ratio</th>  <td>   -0.0439</td> <td>    0.047</td> <td>   -0.927</td> <td> 0.354</td> <td>   -0.137</td> <td>    0.049</td>
</tr>
<tr>
  <th>Quantity_Binned</th>              <td>   -0.0119</td> <td>    0.016</td> <td>   -0.722</td> <td> 0.470</td> <td>   -0.044</td> <td>    0.020</td>
</tr>
<tr>
  <th>Total_Binned</th>                 <td>   -0.0134</td> <td>    0.010</td> <td>   -1.316</td> <td> 0.188</td> <td>   -0.033</td> <td>    0.007</td>
</tr>
<tr>
  <th>Profit_Ratio_x_Price_Binned</th>  <td>   -0.0064</td> <td>    0.028</td> <td>   -0.230</td> <td> 0.818</td> <td>   -0.061</td> <td>    0.048</td>
</tr>
<tr>
  <th>Type_DEBIT</th>                   <td>    6.2803</td> <td>    3.275</td> <td>    1.918</td> <td> 0.055</td> <td>   -0.138</td> <td>   12.699</td>
</tr>
<tr>
  <th>Type_PAYMENT</th>                 <td>    6.1172</td> <td>    3.275</td> <td>    1.868</td> <td> 0.062</td> <td>   -0.302</td> <td>   12.536</td>
</tr>
<tr>
  <th>Type_TRANSFER</th>                <td>    9.1082</td> <td>    4.912</td> <td>    1.854</td> <td> 0.064</td> <td>   -0.520</td> <td>   18.737</td>
</tr>
<tr>
  <th>Order_Region_Caribbean</th>       <td>    0.9262</td> <td>    0.487</td> <td>    1.904</td> <td> 0.057</td> <td>   -0.027</td> <td>    1.880</td>
</tr>
<tr>
  <th>Order_Region_Central Africa</th>  <td>    3.8077</td> <td>    1.950</td> <td>    1.952</td> <td> 0.051</td> <td>   -0.015</td> <td>    7.630</td>
</tr>
<tr>
  <th>Order_Region_Central America</th> <td>    0.9141</td> <td>    0.487</td> <td>    1.878</td> <td> 0.060</td> <td>   -0.040</td> <td>    1.868</td>
</tr>
<tr>
  <th>Order_Region_Central Asia</th>    <td>    0.6395</td> <td>    0.314</td> <td>    2.034</td> <td> 0.042</td> <td>    0.023</td> <td>    1.256</td>
</tr>
<tr>
  <th>Order_Region_East Africa</th>     <td>    3.6876</td> <td>    1.950</td> <td>    1.891</td> <td> 0.059</td> <td>   -0.134</td> <td>    7.510</td>
</tr>
<tr>
  <th>Order_Region_East of USA</th>     <td>   -0.1696</td> <td>    0.131</td> <td>   -1.294</td> <td> 0.196</td> <td>   -0.426</td> <td>    0.087</td>
</tr>
<tr>
  <th>Order_Region_Eastern Asia</th>    <td>    0.3871</td> <td>    0.280</td> <td>    1.382</td> <td> 0.167</td> <td>   -0.162</td> <td>    0.936</td>
</tr>
<tr>
  <th>Order_Region_Eastern Europe</th>  <td>    0.8024</td> <td>    0.394</td> <td>    2.035</td> <td> 0.042</td> <td>    0.030</td> <td>    1.575</td>
</tr>
<tr>
  <th>Order_Region_North Africa</th>    <td>    3.7501</td> <td>    1.950</td> <td>    1.923</td> <td> 0.054</td> <td>   -0.071</td> <td>    7.572</td>
</tr>
<tr>
  <th>Order_Region_Northern Europe</th> <td>    0.8032</td> <td>    0.390</td> <td>    2.057</td> <td> 0.040</td> <td>    0.038</td> <td>    1.569</td>
</tr>
<tr>
  <th>Order_Region_Oceania</th>         <td>    0.5312</td> <td>    0.280</td> <td>    1.899</td> <td> 0.058</td> <td>   -0.017</td> <td>    1.080</td>
</tr>
<tr>
  <th>Order_Region_South America</th>   <td>    0.8002</td> <td>    0.487</td> <td>    1.644</td> <td> 0.100</td> <td>   -0.154</td> <td>    1.754</td>
</tr>
<tr>
  <th>Order_Region_South Asia</th>      <td>    0.4300</td> <td>    0.281</td> <td>    1.530</td> <td> 0.126</td> <td>   -0.121</td> <td>    0.981</td>
</tr>
<tr>
  <th>Order_Region_South of  USA</th>   <td>   -0.1622</td> <td>    0.135</td> <td>   -1.197</td> <td> 0.231</td> <td>   -0.428</td> <td>    0.103</td>
</tr>
<tr>
  <th>Order_Region_Southeast Asia</th>  <td>    0.4673</td> <td>    0.279</td> <td>    1.673</td> <td> 0.094</td> <td>   -0.080</td> <td>    1.015</td>
</tr>
<tr>
  <th>Order_Region_Southern Africa</th> <td>    3.6122</td> <td>    1.951</td> <td>    1.851</td> <td> 0.064</td> <td>   -0.212</td> <td>    7.437</td>
</tr>
<tr>
  <th>Order_Region_Southern Europe</th> <td>    0.6485</td> <td>    0.390</td> <td>    1.663</td> <td> 0.096</td> <td>   -0.116</td> <td>    1.413</td>
</tr>
<tr>
  <th>Order_Region_US Center</th>       <td>   -0.1889</td> <td>    0.131</td> <td>   -1.440</td> <td> 0.150</td> <td>   -0.446</td> <td>    0.068</td>
</tr>
<tr>
  <th>Order_Region_West Africa</th>     <td>    3.5710</td> <td>    1.950</td> <td>    1.832</td> <td> 0.067</td> <td>   -0.251</td> <td>    7.392</td>
</tr>
<tr>
  <th>Order_Region_West Asia</th>       <td>    0.5531</td> <td>    0.286</td> <td>    1.933</td> <td> 0.053</td> <td>   -0.008</td> <td>    1.114</td>
</tr>
<tr>
  <th>Order_Region_West of USA</th>     <td>   -0.0745</td> <td>    0.130</td> <td>   -0.575</td> <td> 0.565</td> <td>   -0.328</td> <td>    0.179</td>
</tr>
<tr>
  <th>Order_Region_Western Europe</th>  <td>    0.6563</td> <td>    0.390</td> <td>    1.682</td> <td> 0.093</td> <td>   -0.108</td> <td>    1.421</td>
</tr>
<tr>
  <th>Market_Europe</th>                <td>    2.9104</td> <td>    1.558</td> <td>    1.868</td> <td> 0.062</td> <td>   -0.143</td> <td>    5.964</td>
</tr>
<tr>
  <th>Market_LATAM</th>                 <td>    2.6405</td> <td>    1.458</td> <td>    1.811</td> <td> 0.070</td> <td>   -0.217</td> <td>    5.498</td>
</tr>
<tr>
  <th>Market_Pacific Asia</th>          <td>    3.0082</td> <td>    1.668</td> <td>    1.804</td> <td> 0.071</td> <td>   -0.260</td> <td>    6.277</td>
</tr>
<tr>
  <th>Market_USCA</th>                  <td>    3.7134</td> <td>    1.951</td> <td>    1.903</td> <td> 0.057</td> <td>   -0.111</td> <td>    7.538</td>
</tr>
<tr>
  <th>Order_Status_CLOSED</th>          <td>    9.1953</td> <td>    4.912</td> <td>    1.872</td> <td> 0.061</td> <td>   -0.432</td> <td>   18.823</td>
</tr>
<tr>
  <th>Order_Status_COMPLETE</th>        <td>    3.0783</td> <td>    1.638</td> <td>    1.880</td> <td> 0.060</td> <td>   -0.132</td> <td>    6.288</td>
</tr>
<tr>
  <th>Order_Status_ON_HOLD</th>         <td>    3.2020</td> <td>    1.637</td> <td>    1.956</td> <td> 0.050</td> <td>   -0.007</td> <td>    6.411</td>
</tr>
<tr>
  <th>Order_Status_PAYMENT_REVIEW</th>  <td>    3.0325</td> <td>    1.639</td> <td>    1.851</td> <td> 0.064</td> <td>   -0.179</td> <td>    6.244</td>
</tr>
<tr>
  <th>Order_Status_PENDING</th>         <td>   -0.0356</td> <td>    0.076</td> <td>   -0.469</td> <td> 0.639</td> <td>   -0.184</td> <td>    0.113</td>
</tr>
<tr>
  <th>Order_Status_PENDING_PAYMENT</th> <td>    3.0846</td> <td>    1.638</td> <td>    1.883</td> <td> 0.060</td> <td>   -0.125</td> <td>    6.295</td>
</tr>
<tr>
  <th>Order_Status_PROCESSING</th>      <td>    0.0255</td> <td>    0.076</td> <td>    0.337</td> <td> 0.736</td> <td>   -0.123</td> <td>    0.174</td>
</tr>
<tr>
  <th>Order_Status_SUSPECTED_FRAUD</th> <td>    0.0588</td> <td>    0.097</td> <td>    0.606</td> <td> 0.544</td> <td>   -0.131</td> <td>    0.249</td>
</tr>
<tr>
  <th>Order_Day_of_Week_Ordinal</th>    <td>   -0.0153</td> <td>    0.005</td> <td>   -3.294</td> <td> 0.001</td> <td>   -0.024</td> <td>   -0.006</td>
</tr>
<tr>
  <th>Shipping_Day_of_Week_Ordinal</th> <td>   -0.0134</td> <td>    0.005</td> <td>   -2.871</td> <td> 0.004</td> <td>   -0.023</td> <td>   -0.004</td>
</tr>
<tr>
  <th>Order_City_Frequency_Encoded</th> <td>   -7.0988</td> <td>    3.688</td> <td>   -1.925</td> <td> 0.054</td> <td>  -14.327</td> <td>    0.129</td>
</tr>
<tr>
  <th>Customer_City_Target_Encoded</th> <td>    1.0225</td> <td>    0.182</td> <td>    5.612</td> <td> 0.000</td> <td>    0.665</td> <td>    1.380</td>
</tr>
<tr>
  <th>Order_Country_Target_Encoded</th> <td>    0.9241</td> <td>    0.101</td> <td>    9.171</td> <td> 0.000</td> <td>    0.727</td> <td>    1.122</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>6290.283</td> <th>  Durbin-Watson:     </th> <td>   1.999</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1276.200</td> 
</tr>
<tr>
  <th>Skew:</th>           <td> 0.123</td>  <th>  Prob(JB):          </th> <td>7.53e-278</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.019</td>  <th>  Cond. No.          </th> <td>1.00e+16</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 2.44e-21. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



The model's R-squared value is low, indicating it only explains a small fraction of the variation in shipping days. Additionally, several variables show insignificant coefficients with high p-values, suggesting they are not contributing to the model. This points to a **potential non-linear relationship** that cannot be adequately captured by the current linear model, and a different modeling approach might be necessary for better accuracy.


```python
X_train.shape
```




    (29934, 59)




```python
numeric_only_df.columns
```




    Index(['Days_for_shipping_(real)', 'Benefit_per_order', 'Sales_per_customer',
           'Category_Id', 'Department_Id', 'Product_Price', 'order_month',
           'order_year', 'shipping_month', 'shipping_year', 'Orders_Made_That_Day',
           'Orders_Shipped_That_Day', 'Price_Binned',
           'Log_Order_Item_Profit_Ratio', 'Quantity_Binned', 'Total_Binned',
           'Profit_Ratio_x_Price_Binned', 'Type_DEBIT', 'Type_PAYMENT',
           'Type_TRANSFER', 'Order_Region_Caribbean',
           'Order_Region_Central Africa', 'Order_Region_Central America',
           'Order_Region_Central Asia', 'Order_Region_East Africa',
           'Order_Region_East of USA', 'Order_Region_Eastern Asia',
           'Order_Region_Eastern Europe', 'Order_Region_North Africa',
           'Order_Region_Northern Europe', 'Order_Region_Oceania',
           'Order_Region_South America', 'Order_Region_South Asia',
           'Order_Region_South of  USA', 'Order_Region_Southeast Asia',
           'Order_Region_Southern Africa', 'Order_Region_Southern Europe',
           'Order_Region_US Center', 'Order_Region_West Africa',
           'Order_Region_West Asia', 'Order_Region_West of USA',
           'Order_Region_Western Europe', 'Market_Europe', 'Market_LATAM',
           'Market_Pacific Asia', 'Market_USCA', 'Order_Status_CLOSED',
           'Order_Status_COMPLETE', 'Order_Status_ON_HOLD',
           'Order_Status_PAYMENT_REVIEW', 'Order_Status_PENDING',
           'Order_Status_PENDING_PAYMENT', 'Order_Status_PROCESSING',
           'Order_Status_SUSPECTED_FRAUD', 'Order_Day_of_Week_Ordinal',
           'Shipping_Day_of_Week_Ordinal', 'Order_City_Frequency_Encoded',
           'Customer_City_Target_Encoded', 'Order_Country_Target_Encoded'],
          dtype='object')



##### 2.  Sklearn's Linear Model


```python
from sklearn.linear_model import LinearRegression

# Split the data into training and testing sets
# - test_size=0.2: 20% of the data will be used for testing
# - random_state=42: Ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Linear Regression model
lr = LinearRegression()

# Fit the model on the training data
model = lr.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

```


```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Evaluate the model's performance using regression metrics

# Mean Absolute Error (MAE) measures the average magnitude of errors in predictions.
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# Mean Squared Error (MSE) measures the average of the squares of the errors.
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# Root Mean Squared Error (RMSE) is the square root of MSE, giving error magnitude in the same units as the target.
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))

# R-squared (R2) shows how well the model's predictions match the actual data, with 1 being perfect prediction.
print("R-squared (R2):", r2_score(y_test, y_pred))
```

    Mean Absolute Error (MAE): 1.358586139597125
    Mean Squared Error (MSE): 2.4372027334649324
    Root Mean Squared Error (RMSE): 1.5611542952139397
    R-squared (R2): 0.04266170122082613
    

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(
    

The model's R-squared value is similarly low, indicating it only explains a small fraction of the variation in shipping days. This points to a **potential non-linear relationship** that cannot be adequately captured by the current linear model, and a different modeling approach should be considered.

##### 3. Logistic Regression classiifier model


```python
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define feature columns by dropping the target column and selecting only numeric columns
X = numeric_only_df.drop(columns=['Days_for_shipping_(real)']).select_dtypes(include=[float, int]) # Feature columns


# Define the target column
y = numeric_only_df['Days_for_shipping_(real)'] # Target column

# Split the data into training and testing sets
# - 30% of the data is used for testing
# - random_state=42 ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```


```python
# Initialize the Logistic Regression model
# - solver='liblinear': Uses the liblinear solver for small datasets or binary classification.
# - multi_class='ovr': Uses one-vs-rest for handling multiple classes.
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# Train the model with the training data
model.fit(X_train, y_train)

```


```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       247
               1       0.00      0.00      0.00       219
               2       0.31      1.00      0.47      3468
               3       0.00      0.00      0.00      1818
               4       0.00      0.00      0.00      1834
               5       0.00      0.00      0.00      1767
               6       0.00      0.00      0.00      1873
    
        accuracy                           0.31     11226
       macro avg       0.04      0.14      0.07     11226
    weighted avg       0.10      0.31      0.15     11226
    
    

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    

The model isn't performing well:

Precision, Recall, F1-Score: It’s failing to predict most classes, with scores of 0.00 for many.
Accuracy: The overall accuracy is just 32%, indicating it's not working effectively.
Averages: Both macro and weighted averages are low, showing poor performance across the board.
Next Steps: The model needs adjustments or better data to improve its predictions for most classes.


```python
# Evaluate the model's performance using regression metrics

# Mean Absolute Error (MAE) measures the average magnitude of errors in predictions.
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# Mean Squared Error (MSE) measures the average of the squares of the errors.
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# Root Mean Squared Error (RMSE) is the square root of MSE, giving error magnitude in the same units as the target.
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))

# R-squared (R2) shows how well the model's predictions match the actual data, with 1 being perfect prediction.
print("R-squared (R2):", r2_score(y_test, y_pred))
```

    Mean Absolute Error (MAE): 1.6917869232139675
    Mean Squared Error (MSE): 5.009086050240513
    Root Mean Squared Error (RMSE): 2.23809875792837
    R-squared (R2): -0.9562073385732774
    

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(
    

The model's R-squared value is also low explaining only a small fraction of the variation in shipping days. This can be a **potential non-linear relationship** or requiredhyperparameter tuning.



```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Assuming you have more than two classes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2,3,4,5,6])  # Modify for the number of classes
n_classes = y_test_bin.shape[1]

# Fit the model
y_score = model.predict_proba(X_test)

# Plot ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.show()

```

##### 4 .Sklearn's Random Forest Classifier

The RandomForestClassifier is a powerful machine learning algorithm used for classification tasks. It works by creating multiple decision trees during training and merging their results to improve accuracy and avoid overfitting. 


```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

```

                  precision    recall  f1-score   support
    
               0       0.96      0.27      0.42       165
               1       0.93      0.26      0.40       151
               2       0.61      0.97      0.75      2276
               3       0.87      0.67      0.76      1250
               4       0.86      0.66      0.74      1233
               5       0.87      0.64      0.74      1170
               6       0.86      0.70      0.77      1239
    
        accuracy                           0.74      7484
       macro avg       0.85      0.60      0.65      7484
    weighted avg       0.79      0.74      0.74      7484
    
    


```python
# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)

```

    Cross-validation scores: [0.54676644 0.50815072 0.71712988 0.58706401 0.45048777]
    


```python
from imblearn.over_sampling import RandomOverSampler

# Handle class imbalance by oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Build the Random Forest model using the balanced data
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)

# Make predictions on the test set and evaluate performance
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Perform 5-fold cross-validation to assess model stability
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)

```

                  precision    recall  f1-score   support
    
               0       0.92      0.49      0.64       165
               1       0.90      0.42      0.58       151
               2       0.70      0.94      0.80      2276
               3       0.84      0.72      0.77      1250
               4       0.82      0.71      0.76      1233
               5       0.84      0.71      0.77      1170
               6       0.82      0.77      0.79      1239
    
        accuracy                           0.78      7484
       macro avg       0.83      0.68      0.73      7484
    weighted avg       0.79      0.78      0.78      7484
    
    Cross-validation scores: [0.53500802 0.54930518 0.74104757 0.59922491 0.4837632 ]
    

the model has an accuracy of 78 %  . this is better but for the Precision, Recall, F1-Score there is more refinement to be done 





```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix with a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


```


    
![png](test%20copy_files/test%20copy_66_0.png)
    



```python
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Assuming the model is multiclass and you have more than 7 classes (from 0 to 6)
# Binarize the test labels for multi-class ROC curve analysis
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])  # Specify classes from 0 to 6
n_classes = y_test_bin.shape[1]

# Get the predicted probabilities for each class
y_score = model.predict_proba(X_test)

# Loop through each class and plot its ROC curve
plt.figure(figsize=(10, 8))  # Set figure size for better clarity
for i in range(n_classes):
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    
    # Compute Area Under the Curve (AUC) for the current class
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Add a diagonal line to represent a random classifier (chance level)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot aesthetics
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')  # Place the legend at the bottom-right corner
plt.grid(True)  # Add grid for better readability
plt.show()

# After evaluating the model with ROC curves, move to regression metrics to assess performance

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Compute Root Mean Squared Error (RMSE), a square root of MSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)

# Compute R-squared (R2) score, which represents the proportion of variance explained by the model
r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)

```


    
![png](test%20copy_files/test%20copy_67_0.png)
    


    Mean Absolute Error (MAE): 0.4722073757349011
    Mean Squared Error (MSE): 1.300106894708712
    Root Mean Squared Error (RMSE): 1.1402223005663028
    R-squared (R2): 0.48931530983389926
    

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(
    


```python
# Loop through each column in the dataframe and print the counts of unique values
for x in numeric_only_df.columns:
    # Print the count of each unique value in the current column
    print(numeric_only_df[x].value_counts())

```

    Days_for_shipping_(real)
    2    11503
    6     6177
    4     6129
    3     6128
    5     5918
    0      819
    1      744
    Name: count, dtype: int64
    Benefit_per_order
    -0.328028    243
     0.615188     70
     0.803831     50
     0.506456     50
     0.285062     49
                ... 
    -0.388289      1
    -0.202004      1
    -0.326063      1
    -0.347810      1
     0.368118      1
    Name: count, Length: 11014, dtype: int64
    Sales_per_customer
    -0.211575    355
     0.097990    339
     0.201179    336
    -0.005198    335
     0.242454    333
                ... 
    -1.661374      1
    -1.553129      1
    -1.359341      1
    -1.336640      1
    -0.054935      1
    Name: count, Length: 2196, dtype: int64
    Category_Id
    48    5738
    17    4723
    46    4477
    18    4334
    24    4182
    43    2536
    9     2361
    29    2134
    45    1697
    35     403
    36     397
    37     365
    40     337
    26     241
    7      237
    13     214
    74     207
    41     201
    33     181
    32     177
    75     146
    63     136
    73     135
    3      126
    62     118
    76     117
    67     107
    72      98
    30      97
    66      94
    65      94
    44      88
    68      84
    12      81
    31      74
    59      72
    71      72
    38      69
    69      66
    6       64
    5       60
    11      53
    61      48
    60      41
    70      40
    2       30
    34      24
    10      19
    16      11
    4        8
    64       4
    Name: count, dtype: int64
    Department_Id
    7     14743
    4      9485
    5      6557
    3      2739
    6      2409
    2       660
    9       373
    10      216
    11       98
    8        72
    12       66
    Name: count, dtype: int64
    Product_Price
     0.665786     5742
    -0.632430     4778
    -0.725252     4477
     0.016678     4384
    -0.725067     4182
                  ... 
     0.563876        9
    -0.169152        8
     12.720735       4
     4.374972        4
     8.084159        1
    Name: count, Length: 74, dtype: int64
    order_month
    1     3700
    8     3548
    9     3444
    5     3388
    7     3338
    3     3224
    6     3219
    4     3024
    2     2913
    12    2581
    10    2580
    11    2459
    Name: count, dtype: int64
    order_year
    2015    12560
    2016    12232
    2017    12150
    2018      476
    Name: count, dtype: int64
    shipping_month
    8     3556
    1     3545
    5     3426
    9     3404
    7     3302
    3     3259
    6     3241
    2     2992
    4     2953
    10    2788
    12    2568
    11    2384
    Name: count, dtype: int64
    shipping_year
    2015    12452
    2017    12238
    2016    12199
    2018      529
    Name: count, dtype: int64
    Orders_Made_That_Day
    124    1237
    119    1024
    121     952
    115     949
    111     911
           ... 
    39       17
    59       15
    40       11
    41       10
    37        8
    Name: count, Length: 120, dtype: int64
    Orders_Shipped_That_Day
    118    966
    125    912
    112    899
    120    869
    114    849
          ... 
    69      10
    26       9
    24       6
    8        3
    16       2
    Name: count, Length: 138, dtype: int64
    Price_Binned
    3    13913
    0    10351
    1     7808
    2     5346
    Name: count, dtype: int64
    Log_Order_Item_Profit_Ratio
     0.392042    1943
     0.300105    1738
     0.292670    1421
     0.231112    1401
     0.385262    1381
                 ... 
    -1.966113       7
    -0.616186       5
    -2.302585       5
    -0.673345       3
    -2.659260       2
    Name: count, Length: 125, dtype: int64
    Quantity_Binned
    0    20608
    4     4490
    3     4200
    2     4095
    1     4025
    Name: count, dtype: int64
    Total_Binned
    3    10300
    0     9703
    1     8926
    2     8489
    Name: count, dtype: int64
    Profit_Ratio_x_Price_Binned
    -0.284298    448
    -0.217627    418
    -0.212236    343
    -0.279381    320
    -0.167596    306
                ... 
     0.031126      1
    -0.010168      1
     0.228031      1
    -0.059545      1
    -3.516262      1
    Name: count, Length: 2859, dtype: int64
    Type_DEBIT
    0    19815
    1    17603
    Name: count, dtype: int64
    Type_PAYMENT
    0    28760
    1     8658
    Name: count, dtype: int64
    Type_TRANSFER
    0    29142
    1     8276
    Name: count, dtype: int64
    Order_Region_Caribbean
    0    35571
    1     1847
    Name: count, dtype: int64
    Order_Region_Central Africa
    0    37072
    1      346
    Name: count, dtype: int64
    Order_Region_Central America
    0    31357
    1     6061
    Name: count, dtype: int64
    Order_Region_Central Asia
    0    37299
    1      119
    Name: count, dtype: int64
    Order_Region_East Africa
    0    37011
    1      407
    Name: count, dtype: int64
    Order_Region_East of USA
    0    36030
    1     1388
    Name: count, dtype: int64
    Order_Region_Eastern Asia
    0    36060
    1     1358
    Name: count, dtype: int64
    Order_Region_Eastern Europe
    0    36582
    1      836
    Name: count, dtype: int64
    Order_Region_North Africa
    0    36768
    1      650
    Name: count, dtype: int64
    Order_Region_Northern Europe
    0    35366
    1     2052
    Name: count, dtype: int64
    Order_Region_Oceania
    0    35656
    1     1762
    Name: count, dtype: int64
    Order_Region_South America
    0    34183
    1     3235
    Name: count, dtype: int64
    Order_Region_South Asia
    0    35839
    1     1579
    Name: count, dtype: int64
    Order_Region_South of  USA
    0    36603
    1      815
    Name: count, dtype: int64
    Order_Region_Southeast Asia
    0    35731
    1     1687
    Name: count, dtype: int64
    Order_Region_Southern Africa
    0    37178
    1      240
    Name: count, dtype: int64
    Order_Region_Southern Europe
    0    35397
    1     2021
    Name: count, dtype: int64
    Order_Region_US Center
    0    36247
    1     1171
    Name: count, dtype: int64
    Order_Region_West Africa
    0    36657
    1      761
    Name: count, dtype: int64
    Order_Region_West Asia
    0    36141
    1     1277
    Name: count, dtype: int64
    Order_Region_West of USA
    0    35810
    1     1608
    Name: count, dtype: int64
    Order_Region_Western Europe
    0    31428
    1     5990
    Name: count, dtype: int64
    Market_Europe
    0    26519
    1    10899
    Name: count, dtype: int64
    Market_LATAM
    0    26275
    1    11143
    Name: count, dtype: int64
    Market_Pacific Asia
    0    29636
    1     7782
    Name: count, dtype: int64
    Market_USCA
    0    32228
    1     5190
    Name: count, dtype: int64
    Order_Status_CLOSED
    0    34537
    1     2881
    Name: count, dtype: int64
    Order_Status_COMPLETE
    0    22328
    1    15090
    Name: count, dtype: int64
    Order_Status_ON_HOLD
    0    34905
    1     2513
    Name: count, dtype: int64
    Order_Status_PAYMENT_REVIEW
    0    37059
    1      359
    Name: count, dtype: int64
    Order_Status_PENDING
    0    34045
    1     3373
    Name: count, dtype: int64
    Order_Status_PENDING_PAYMENT
    0    29119
    1     8299
    Name: count, dtype: int64
    Order_Status_PROCESSING
    0    33833
    1     3585
    Name: count, dtype: int64
    Order_Status_SUSPECTED_FRAUD
    0    36739
    1      679
    Name: count, dtype: int64
    Order_Day_of_Week_Ordinal
    0    5489
    3    5400
    5    5349
    4    5342
    1    5325
    2    5275
    6    5238
    Name: count, dtype: int64
    Shipping_Day_of_Week_Ordinal
    6    5431
    5    5356
    4    5355
    3    5333
    2    5331
    1    5321
    0    5291
    Name: count, dtype: int64
    Order_City_Frequency_Encoded
    0.000098    502
    0.012815    496
    0.000163    453
    0.011834    438
    0.000270    413
               ... 
    0.000793     33
    0.000785     31
    0.001242     29
    0.000809     29
    0.001144     21
    Name: count, Length: 246, dtype: int64
    Customer_City_Target_Encoded
    3.563726    36104
    3.350283      173
    3.834481      142
    3.427279      104
    3.901857       90
    3.256116       84
    3.447105       84
    3.670359       78
    3.954839       70
    3.755274       66
    3.651294       58
    3.080013       56
    4.161185       53
    3.628478       51
    3.312892       39
    3.597500       37
    3.721497       31
    3.529012       24
    3.491140       24
    3.452014       21
    3.196809       15
    3.423334       14
    Name: count, dtype: int64
    Order_Country_Target_Encoded
    3.466679    4982
    3.497723    2977
    3.509363    2891
    3.487221    2067
    3.554651    1722
                ... 
    3.545848       2
    2.968057       2
    3.789970       1
    3.620327       1
    3.415740       1
    Name: count, Length: 146, dtype: int64
    


```python
import xgboost as xgb
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

model = xgb.XGBClassifier()
model.fit(X_resampled, y_resampled)

```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, objective=&#x27;multi:softprob&#x27;, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;XGBClassifier<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, objective=&#x27;multi:softprob&#x27;, ...)</pre></div> </div></div></div></div>




```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Benefit_per_order</th>
      <th>Sales_per_customer</th>
      <th>Category_Id</th>
      <th>Department_Id</th>
      <th>Product_Price</th>
      <th>order_month</th>
      <th>order_year</th>
      <th>shipping_month</th>
      <th>shipping_year</th>
      <th>Orders_Made_That_Day</th>
      <th>...</th>
      <th>Order_Status_PAYMENT_REVIEW</th>
      <th>Order_Status_PENDING</th>
      <th>Order_Status_PENDING_PAYMENT</th>
      <th>Order_Status_PROCESSING</th>
      <th>Order_Status_SUSPECTED_FRAUD</th>
      <th>Order_Day_of_Week_Ordinal</th>
      <th>Shipping_Day_of_Week_Ordinal</th>
      <th>Order_City_Frequency_Encoded</th>
      <th>Customer_City_Target_Encoded</th>
      <th>Order_Country_Target_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105554</th>
      <td>-0.098905</td>
      <td>-1.243564</td>
      <td>46</td>
      <td>7</td>
      <td>-0.725252</td>
      <td>5</td>
      <td>2017</td>
      <td>5</td>
      <td>2017</td>
      <td>111</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0.000433</td>
      <td>3.563726</td>
      <td>3.509363</td>
    </tr>
    <tr>
      <th>123104</th>
      <td>1.454519</td>
      <td>2.140813</td>
      <td>9</td>
      <td>3</td>
      <td>-0.261511</td>
      <td>7</td>
      <td>2016</td>
      <td>7</td>
      <td>2016</td>
      <td>85</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.004961</td>
      <td>3.563726</td>
      <td>3.466679</td>
    </tr>
    <tr>
      <th>10849</th>
      <td>0.079651</td>
      <td>0.809888</td>
      <td>43</td>
      <td>7</td>
      <td>1.592989</td>
      <td>2</td>
      <td>2015</td>
      <td>2</td>
      <td>2015</td>
      <td>121</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.010085</td>
      <td>3.563726</td>
      <td>3.464115</td>
    </tr>
    <tr>
      <th>13119</th>
      <td>-0.076504</td>
      <td>2.202933</td>
      <td>45</td>
      <td>7</td>
      <td>2.520286</td>
      <td>9</td>
      <td>2015</td>
      <td>9</td>
      <td>2015</td>
      <td>103</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.002109</td>
      <td>3.427279</td>
      <td>3.423427</td>
    </tr>
    <tr>
      <th>26684</th>
      <td>-1.239804</td>
      <td>1.831454</td>
      <td>45</td>
      <td>7</td>
      <td>2.520286</td>
      <td>12</td>
      <td>2015</td>
      <td>12</td>
      <td>2015</td>
      <td>119</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0.002321</td>
      <td>3.563726</td>
      <td>3.355634</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       165
               1       1.00      1.00      1.00       151
               2       1.00      1.00      1.00      2276
               3       1.00      1.00      1.00      1250
               4       1.00      1.00      1.00      1233
               5       1.00      1.00      1.00      1170
               6       1.00      1.00      1.00      1239
    
        accuracy                           1.00      7484
       macro avg       1.00      1.00      1.00      7484
    weighted avg       1.00      1.00      1.00      7484
    
    Cross-validation scores: [1.         1.         1.         0.99986636 1.        ]
    


```python
# Evaluate the model with regression metrics
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
print("R-squared (R2):", r2_score(y_test, y_pred))
```

    Mean Absolute Error (MAE): 0.0
    Mean Squared Error (MSE): 0.0
    Root Mean Squared Error (RMSE): 0.0
    R-squared (R2): 1.0
    

    c:\Users\SEVEN\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(
    


```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict using the model
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

```


    
![png](test%20copy_files/test%20copy_73_0.png)
    



```python
import xgboost as xgb
import matplotlib.pyplot as plt

# Get the feature importance from the model
importance = model.get_booster().get_score(importance_type='weight')

# Sort the features by importance and select the top 15
sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

# Convert the sorted importance to a list of tuples for easier plotting
sorted_importance_list = list(sorted_importance.items())

# Plot the importance for top 15 features
xgb.plot_importance(model, importance_type='weight', max_num_features=20)
plt.title('Top 20 Feature Importances')
plt.show()

```


    
![png](test%20copy_files/test%20copy_74_0.png)
    



```python
sorted_importance_list
```




    [('Order_Day_of_Week_Ordinal', 3820.0),
     ('Shipping_Day_of_Week_Ordinal', 3718.0),
     ('Orders_Shipped_That_Day', 1312.0),
     ('Orders_Made_That_Day', 1209.0),
     ('Order_City_Frequency_Encoded', 897.0),
     ('Order_Country_Target_Encoded', 636.0),
     ('Sales_per_customer', 407.0),
     ('Benefit_per_order', 395.0),
     ('order_month', 372.0),
     ('Profit_Ratio_x_Price_Binned', 339.0),
     ('Log_Order_Item_Profit_Ratio', 283.0),
     ('shipping_month', 268.0),
     ('Category_Id', 262.0),
     ('Customer_City_Target_Encoded', 140.0),
     ('Type_DEBIT', 134.0),
     ('Product_Price', 109.0),
     ('Market_Pacific Asia', 81.0),
     ('order_year', 80.0),
     ('Order_Status_PENDING', 63.0),
     ('Type_TRANSFER', 61.0)]




```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Assuming you have more than two classes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2,3,4,5,6])  # Modify for the number of classes
n_classes = y_test_bin.shape[1]

# Fit the model
y_score = model.predict_proba(X_test)

# Plot ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.show()

```


    
![png](test%20copy_files/test%20copy_76_0.png)
    



```python
# Plot SHAP summary for each class in a multiclass model
for i in range(shap_values.shape[2]):
    # Extract SHAP values for class i
    class_shap_values = shap_values[:, :, i]  
    
    # Create a figure for better visualization, setting size for clarity
    plt.figure(figsize=(10, 6))
    
    # Plot the SHAP summary for the current class with a title indicating the class number
    shap.summary_plot(class_shap_values, X_test, title=f"SHAP Summary Plot for Class {i}")
    
    # Display the plot
    plt.show()

```


    
![png](test%20copy_files/test%20copy_77_0.png)
    



    
![png](test%20copy_files/test%20copy_77_1.png)
    



    
![png](test%20copy_files/test%20copy_77_2.png)
    



    
![png](test%20copy_files/test%20copy_77_3.png)
    



    
![png](test%20copy_files/test%20copy_77_4.png)
    



    
![png](test%20copy_files/test%20copy_77_5.png)
    



    
![png](test%20copy_files/test%20copy_77_6.png)
    

