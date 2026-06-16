---
output:
  word_document: default
  html_document: default
---
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
```


```python
employee = pd.read_csv('/Users/shraddha/Desktop/MPSA - SHRADDHA GUPTE/Fall 2024 Q3/Data Mining Applications - ALY 6040/Module 1/Extended_Employee_Performance_and_Productivity_Data.csv')
```


```python
summary = employee.describe()
info = employee.info()

print(summary)
print(info)
employee_str = str(employee)
employee_str
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 20 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   Employee_ID                  100000 non-null  int64  
     1   Department                   100000 non-null  object 
     2   Gender                       100000 non-null  object 
     3   Age                          100000 non-null  int64  
     4   Job_Title                    100000 non-null  object 
     5   Hire_Date                    100000 non-null  object 
     6   Years_At_Company             100000 non-null  int64  
     7   Education_Level              100000 non-null  object 
     8   Performance_Score            100000 non-null  int64  
     9   Monthly_Salary               100000 non-null  float64
     10  Work_Hours_Per_Week          100000 non-null  int64  
     11  Projects_Handled             100000 non-null  int64  
     12  Overtime_Hours               100000 non-null  int64  
     13  Sick_Days                    100000 non-null  int64  
     14  Remote_Work_Frequency        100000 non-null  int64  
     15  Team_Size                    100000 non-null  int64  
     16  Training_Hours               100000 non-null  int64  
     17  Promotions                   100000 non-null  int64  
     18  Employee_Satisfaction_Score  100000 non-null  float64
     19  Resigned                     100000 non-null  bool   
    dtypes: bool(1), float64(2), int64(12), object(5)
    memory usage: 14.6+ MB
             Employee_ID            Age  Years_At_Company  Performance_Score  \
    count  100000.000000  100000.000000     100000.000000      100000.000000   
    mean    50000.500000      41.029410          4.476070           2.995430   
    std     28867.657797      11.244121          2.869336           1.414726   
    min         1.000000      22.000000          0.000000           1.000000   
    25%     25000.750000      31.000000          2.000000           2.000000   
    50%     50000.500000      41.000000          4.000000           3.000000   
    75%     75000.250000      51.000000          7.000000           4.000000   
    max    100000.000000      60.000000         10.000000           5.000000   
    
           Monthly_Salary  Work_Hours_Per_Week  Projects_Handled  Overtime_Hours  \
    count   100000.000000        100000.000000     100000.000000   100000.000000   
    mean      6403.211000            44.956950         24.431170       14.514930   
    std       1372.508717             8.942003         14.469584        8.664026   
    min       3850.000000            30.000000          0.000000        0.000000   
    25%       5250.000000            37.000000         12.000000        7.000000   
    50%       6500.000000            45.000000         24.000000       15.000000   
    75%       7500.000000            53.000000         37.000000       22.000000   
    max       9000.000000            60.000000         49.000000       29.000000   
    
               Sick_Days  Remote_Work_Frequency      Team_Size  Training_Hours  \
    count  100000.000000          100000.000000  100000.000000   100000.000000   
    mean        7.008550              50.090500      10.013560       49.506060   
    std         4.331591              35.351157       5.495405       28.890383   
    min         0.000000               0.000000       1.000000        0.000000   
    25%         3.000000              25.000000       5.000000       25.000000   
    50%         7.000000              50.000000      10.000000       49.000000   
    75%        11.000000              75.000000      15.000000       75.000000   
    max        14.000000             100.000000      19.000000       99.000000   
    
              Promotions  Employee_Satisfaction_Score  
    count  100000.000000                100000.000000  
    mean        0.999720                     2.999088  
    std         0.815872                     1.150719  
    min         0.000000                     1.000000  
    25%         0.000000                     2.010000  
    50%         1.000000                     3.000000  
    75%         2.000000                     3.990000  
    max         2.000000                     5.000000  
    None





    '       Employee_ID        Department  Gender  Age   Job_Title  \\\n0                1                IT    Male   55  Specialist   \n1                2           Finance    Male   29   Developer   \n2                3           Finance    Male   55  Specialist   \n3                4  Customer Support  Female   48     Analyst   \n4                5       Engineering  Female   36     Analyst   \n...            ...               ...     ...  ...         ...   \n99995        99996           Finance    Male   27  Technician   \n99996        99997                IT  Female   36  Consultant   \n99997        99998        Operations    Male   53     Analyst   \n99998        99999                HR  Female   22  Consultant   \n99999       100000           Finance  Female   43     Analyst   \n\n                        Hire_Date  Years_At_Company Education_Level  \\\n0      2022-01-19 08:03:05.556036                 2     High School   \n1      2024-04-18 08:03:05.556036                 0     High School   \n2      2015-10-26 08:03:05.556036                 8     High School   \n3      2016-10-22 08:03:05.556036                 7        Bachelor   \n4      2021-07-23 08:03:05.556036                 3        Bachelor   \n...                           ...               ...             ...   \n99995  2022-12-07 08:03:05.556036                 1        Bachelor   \n99996  2018-07-24 08:03:05.556036                 6          Master   \n99997  2015-11-24 08:03:05.556036                 8     High School   \n99998  2015-08-03 08:03:05.556036                 9     High School   \n99999  2024-03-04 08:03:05.556036                 0             PhD   \n\n       Performance_Score  Monthly_Salary  Work_Hours_Per_Week  \\\n0                      5          6750.0                   33   \n1                      5          7500.0                   34   \n2                      3          5850.0                   37   \n3                      2          4800.0                   52   \n4                      2          4800.0                   38   \n...                  ...             ...                  ...   \n99995                  4          4900.0                   55   \n99996                  5          8250.0                   39   \n99997                  2          4800.0                   31   \n99998                  5          8250.0                   35   \n99999                  1          4400.0                   51   \n\n       Projects_Handled  Overtime_Hours  Sick_Days  Remote_Work_Frequency  \\\n0                    32              22          2                      0   \n1                    34              13         14                    100   \n2                    27               6          3                     50   \n3                    10              28         12                    100   \n4                    11              29         13                    100   \n...                 ...             ...        ...                    ...   \n99995                46               5          3                     75   \n99996                35               7          0                      0   \n99997                13               6          5                      0   \n99998                43              10          1                     75   \n99999                43              27         11                     75   \n\n       Team_Size  Training_Hours  Promotions  Employee_Satisfaction_Score  \\\n0             14              66           0                         2.63   \n1             12              61           2                         1.72   \n2             10               1           0                         3.17   \n3             10               0           1                         1.86   \n4             15               9           1                         1.25   \n...          ...             ...         ...                          ...   \n99995         16              48           2                         1.28   \n99996         10              77           1                         3.48   \n99997          5              87           1                         2.60   \n99998          2              31           1                         3.10   \n99999         13              45           1                         2.64   \n\n       Resigned  \n0         False  \n1         False  \n2         False  \n3         False  \n4         False  \n...         ...  \n99995     False  \n99996      True  \n99997     False  \n99998     False  \n99999     False  \n\n[100000 rows x 20 columns]'




```python
missing_values = employee.isnull().sum()
print("Missing values per column:")
print(missing_values)

```

    Missing values per column:
    Employee_ID                    0
    Department                     0
    Gender                         0
    Age                            0
    Job_Title                      0
    Hire_Date                      0
    Years_At_Company               0
    Education_Level                0
    Performance_Score              0
    Monthly_Salary                 0
    Work_Hours_Per_Week            0
    Projects_Handled               0
    Overtime_Hours                 0
    Sick_Days                      0
    Remote_Work_Frequency          0
    Team_Size                      0
    Training_Hours                 0
    Promotions                     0
    Employee_Satisfaction_Score    0
    Resigned                       0
    dtype: int64



```python
df = pd.DataFrame(employee)

# Basic exploration
print(df.info())
print(df.describe())

# number of rows with missing values
num_missing_rows = len(employee) - employee.dropna().shape[0]


```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 20 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   Employee_ID                  100000 non-null  int64  
     1   Department                   100000 non-null  object 
     2   Gender                       100000 non-null  object 
     3   Age                          100000 non-null  int64  
     4   Job_Title                    100000 non-null  object 
     5   Hire_Date                    100000 non-null  object 
     6   Years_At_Company             100000 non-null  int64  
     7   Education_Level              100000 non-null  object 
     8   Performance_Score            100000 non-null  int64  
     9   Monthly_Salary               100000 non-null  float64
     10  Work_Hours_Per_Week          100000 non-null  int64  
     11  Projects_Handled             100000 non-null  int64  
     12  Overtime_Hours               100000 non-null  int64  
     13  Sick_Days                    100000 non-null  int64  
     14  Remote_Work_Frequency        100000 non-null  int64  
     15  Team_Size                    100000 non-null  int64  
     16  Training_Hours               100000 non-null  int64  
     17  Promotions                   100000 non-null  int64  
     18  Employee_Satisfaction_Score  100000 non-null  float64
     19  Resigned                     100000 non-null  bool   
    dtypes: bool(1), float64(2), int64(12), object(5)
    memory usage: 14.6+ MB
    None
             Employee_ID            Age  Years_At_Company  Performance_Score  \
    count  100000.000000  100000.000000     100000.000000      100000.000000   
    mean    50000.500000      41.029410          4.476070           2.995430   
    std     28867.657797      11.244121          2.869336           1.414726   
    min         1.000000      22.000000          0.000000           1.000000   
    25%     25000.750000      31.000000          2.000000           2.000000   
    50%     50000.500000      41.000000          4.000000           3.000000   
    75%     75000.250000      51.000000          7.000000           4.000000   
    max    100000.000000      60.000000         10.000000           5.000000   
    
           Monthly_Salary  Work_Hours_Per_Week  Projects_Handled  Overtime_Hours  \
    count   100000.000000        100000.000000     100000.000000   100000.000000   
    mean      6403.211000            44.956950         24.431170       14.514930   
    std       1372.508717             8.942003         14.469584        8.664026   
    min       3850.000000            30.000000          0.000000        0.000000   
    25%       5250.000000            37.000000         12.000000        7.000000   
    50%       6500.000000            45.000000         24.000000       15.000000   
    75%       7500.000000            53.000000         37.000000       22.000000   
    max       9000.000000            60.000000         49.000000       29.000000   
    
               Sick_Days  Remote_Work_Frequency      Team_Size  Training_Hours  \
    count  100000.000000          100000.000000  100000.000000   100000.000000   
    mean        7.008550              50.090500      10.013560       49.506060   
    std         4.331591              35.351157       5.495405       28.890383   
    min         0.000000               0.000000       1.000000        0.000000   
    25%         3.000000              25.000000       5.000000       25.000000   
    50%         7.000000              50.000000      10.000000       49.000000   
    75%        11.000000              75.000000      15.000000       75.000000   
    max        14.000000             100.000000      19.000000       99.000000   
    
              Promotions  Employee_Satisfaction_Score  
    count  100000.000000                100000.000000  
    mean        0.999720                     2.999088  
    std         0.815872                     1.150719  
    min         0.000000                     1.000000  
    25%         0.000000                     2.010000  
    50%         1.000000                     3.000000  
    75%         2.000000                     3.990000  
    max         2.000000                     5.000000  



```python
# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Countplot for categorical variables
plt.figure(figsize=(14, 6))
sns.countplot(y='Department', data=df, order=df['Department'].value_counts().index)
plt.title('Employee Count by Department')
plt.show()
```


    
![png](output_5_0.png)
    



```python
# 2. Boxplot of Monthly Salary by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Monthly_Salary', data=df)
plt.title('Monthly Salary Distribution by Gender')
plt.show()
```


    
![png](output_6_0.png)
    



```python
# 3. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()


```


    
![png](output_7_0.png)
    



```python

# 4. Average Employee Satisfaction by Performance Score
plt.figure(figsize=(10, 6))
sns.barplot(x='Performance_Score', y='Employee_Satisfaction_Score', data=df)
plt.title('Average Satisfaction Score by Performance Score')
plt.show()
```


    
![png](output_8_0.png)
    



```python
# 5. Pairplot for numeric variables (to see pairwise relationships)
sns.pairplot(df[['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Performance_Score']])
plt.show()

# Summary statistics for key columns
summary_stats = df[['Age', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Employee_Satisfaction_Score']].describe()
print(summary_stats)

```


    
![png](output_9_0.png)
    


                     Age  Monthly_Salary  Work_Hours_Per_Week  Projects_Handled  \
    count  100000.000000   100000.000000        100000.000000     100000.000000   
    mean       41.029410     6403.211000            44.956950         24.431170   
    std        11.244121     1372.508717             8.942003         14.469584   
    min        22.000000     3850.000000            30.000000          0.000000   
    25%        31.000000     5250.000000            37.000000         12.000000   
    50%        41.000000     6500.000000            45.000000         24.000000   
    75%        51.000000     7500.000000            53.000000         37.000000   
    max        60.000000     9000.000000            60.000000         49.000000   
    
           Employee_Satisfaction_Score  
    count                100000.000000  
    mean                      2.999088  
    std                       1.150719  
    min                       1.000000  
    25%                       2.010000  
    50%                       3.000000  
    75%                       3.990000  
    max                       5.000000  



```python
#Data Prepocessing  
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Encoding categorical variables
onehot_encoder = OneHotEncoder(drop='first')

# Normalizing/Standardizing numerical features
scaler = StandardScaler()
```


```python
# Define columns that are numerical and categorical
numeric_cols = employee.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = employee.select_dtypes(include=['object', 'category']).columns
# Imputing missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')
```


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Update OneHotEncoder to return dense output
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', numeric_imputer), ('scaler', scaler)]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', categorical_imputer), ('onehot', onehot_encoder)]), categorical_cols)
    ])

# Apply preprocessing
data_preprocessed = preprocessor.fit_transform(employee)

# Extract column names for the transformed data
encoded_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
columns = list(numeric_cols) + list(encoded_columns)

# Convert preprocessed data back into a DataFrame
New_emp = pd.DataFrame(data_preprocessed, columns=columns)

# Check the preprocessed DataFrame
print(New_emp.head())

```

       Employee_ID       Age  Years_At_Company  Performance_Score  Monthly_Salary  \
    0    -1.732033  1.242486         -0.862946           1.416939        0.252669   
    1    -1.731999 -1.069845         -1.559975           1.416939        0.799117   
    2    -1.731964  1.242486          1.228140           0.003230       -0.403068   
    3    -1.731930  0.619935          0.879626          -0.703624       -1.168094   
    4    -1.731895 -0.447295         -0.514432          -0.703624       -1.168094   
    
       Work_Hours_Per_Week  Projects_Handled  Overtime_Hours  Sick_Days  \
    0            -1.337174          0.523088        0.863929  -1.156290   
    1            -1.225341          0.661310       -0.174854   1.614068   
    2            -0.889844          0.177534       -0.982796  -0.925427   
    3             0.787641         -0.997350        1.556452   1.152342   
    4            -0.778012         -0.928239        1.671872   1.383205   
    
       Remote_Work_Frequency  ...  Hire_Date_2024-08-29 08:03:05.556036  \
    0              -1.416948  ...                                   0.0   
    1               1.411828  ...                                   0.0   
    2              -0.002560  ...                                   0.0   
    3               1.411828  ...                                   0.0   
    4               1.411828  ...                                   0.0   
    
       Hire_Date_2024-08-30 08:03:05.556036  Hire_Date_2024-08-31 08:03:05.556036  \
    0                                   0.0                                   0.0   
    1                                   0.0                                   0.0   
    2                                   0.0                                   0.0   
    3                                   0.0                                   0.0   
    4                                   0.0                                   0.0   
    
       Hire_Date_2024-09-01 08:03:05.556036  Hire_Date_2024-09-02 08:03:05.556036  \
    0                                   0.0                                   0.0   
    1                                   0.0                                   0.0   
    2                                   0.0                                   0.0   
    3                                   0.0                                   0.0   
    4                                   0.0                                   0.0   
    
       Hire_Date_2024-09-03 08:03:05.556036  Education_Level_Bachelor  \
    0                                   0.0                       0.0   
    1                                   0.0                       0.0   
    2                                   0.0                       0.0   
    3                                   0.0                       1.0   
    4                                   0.0                       1.0   
    
       Education_Level_High School  Education_Level_Master  Education_Level_PhD  
    0                          1.0                     0.0                  0.0  
    1                          1.0                     0.0                  0.0  
    2                          1.0                     0.0                  0.0  
    3                          0.0                     0.0                  0.0  
    4                          0.0                     0.0                  0.0  
    
    [5 rows x 3687 columns]



```python
#Feature Selection and Engineering 
selector = VarianceThreshold(threshold=0.01)
New_emp_sel = selector.fit_transform(New_emp)

# Check the shape of the data after feature selection
print(f"Original shape: {New_emp.shape}, After feature selection: {New_emp_sel.shape}")
```

    Original shape: (100000, 3687), After feature selection: (100000, 37)



```python
New_emp_pca = pca.fit_transform(New_emp_sel)

# Check the variance explained again
explained_variance_new = pca.explained_variance_ratio_
print(f"Explained variance by each component after feature engineering: {explained_variance_new}")
```

    Explained variance by each component after feature engineering: [0.09156115 0.06012718 0.05989671 0.05958085 0.05953292 0.05927991
     0.05923109 0.05905135 0.05889702 0.05874557 0.05843503 0.05833737
     0.05820732 0.03502375 0.02837415 0.02273594 0.01107275 0.00855055
     0.00844462 0.00843428 0.00840327 0.00839576 0.00662271 0.00661346]



```python
# Preparing the Data for Modelling 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
New_emp = pd.get_dummies(New_emp, columns=['Work_Hours_Per_Week', 'Projects_Handled', 'Remote_Work_Frequency'], drop_first=True)

X = New_emp.drop(['Performance_Score', 'Employee_ID', 'Monthly_Salary'], axis=1) 
y = New_emp['Performance_Score']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
#Modelling - Performance Analysis

#1. Linear Regression Model 

# Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Mean Squared Error: {mse_lr}")
print(f"Linear Regression R-squared: {r2_lr}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Perfect prediction line
plt.xlabel("True Performance Score")
plt.ylabel("Predicted Performance Score")
plt.title("Linear Regression: True vs Predicted Performance Score")
plt.show()

```

    Linear Regression Mean Squared Error: 1.058242958407331
    Linear Regression R-squared: -0.05970525142425265



    
![png](output_16_1.png)
    



```python
#2. Decision Tree Regressor

# Create and train the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree Mean Squared Error: {mse_dt}")
print(f"Decision Tree R-squared: {r2_dt}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel("True Performance Score")
plt.ylabel("Predicted Performance Score")
plt.title("Decision Tree: True vs Predicted Performance Score")
plt.show()

```

    Decision Tree Mean Squared Error: 2.0253361141217
    Decision Tree R-squared: -1.0281347482473233



    
![png](output_17_1.png)
    



```python
print(New_emp.columns)
print(employee.columns)
```

    Index(['Employee_ID', 'Age', 'Years_At_Company', 'Performance_Score',
           'Monthly_Salary', 'Overtime_Hours', 'Sick_Days', 'Team_Size',
           'Training_Hours', 'Promotions',
           ...
           'Projects_Handled_1.3524181537752231',
           'Projects_Handled_1.421528987370038',
           'Projects_Handled_1.4906398209648533',
           'Projects_Handled_1.5597506545596682',
           'Projects_Handled_1.6288614881544832',
           'Projects_Handled_1.6979723217492981',
           'Remote_Work_Frequency_-0.7097540088343408',
           'Remote_Work_Frequency_-0.0025600421593633817',
           'Remote_Work_Frequency_0.7046339245156141',
           'Remote_Work_Frequency_1.4118278911905915'],
          dtype='object', length=3767)
    Index(['Employee_ID', 'Department', 'Gender', 'Age', 'Job_Title', 'Hire_Date',
           'Years_At_Company', 'Education_Level', 'Performance_Score',
           'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled',
           'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size',
           'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score',
           'Resigned'],
          dtype='object')



```python
#3. KMeans Clusterring Model 
from sklearn.cluster import KMeans

selected_features = ['Team_Size', 'Training_Hours']
employee = pd.get_dummies(employee, columns=['Job_Title'], drop_first=True)

# Scaling the data to normalize the range of values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(employee[selected_features])

# Define the number of clusters (3 for high, medium, low performers)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model
kmeans.fit(X_scaled)

# Get cluster labels
employee['Performance_Cluster'] = kmeans.labels_

# Analyze the distribution of clusters
print(employee['Performance_Cluster'].value_counts())

# Visualizing the clusters (2D Plot for simplicity)
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=employee['Performance_Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Employees by Team Size and Training Hours')
plt.xlabel('Team Size (scaled)')
plt.ylabel('Training Hours (scaled)')
plt.show()
```

    Performance_Cluster
    1    37451
    0    32325
    2    30224
    Name: count, dtype: int64



    
![png](output_19_1.png)
    



```python
# Model Comparision
# Compare Linear Regression and Decision Tree results
print(f"Linear Regression R-squared: {r2_lr}")
print(f"Decision Tree R-squared: {r2_dt}")

print(f"Linear Regression Mean Squared Error: {mse_lr}")
print(f"Decision Tree Mean Squared Error: {mse_dt}")

print(F"Perfromance Cluster: {0} High = 32325")
print(F"Perfromance Cluster: {1} Medium = 37451")
print(F"Perfromance Cluster: {2 } Low = 30224")
```

    Linear Regression R-squared: -0.05970525142425265
    Decision Tree R-squared: -1.0281347482473233
    Linear Regression Mean Squared Error: 1.058242958407331
    Decision Tree Mean Squared Error: 2.0253361141217
    Perfromance Cluster: 0 High = 32325
    Perfromance Cluster: 1 Medium = 37451
    Perfromance Cluster: 2 Low = 30224



```python
# Modelling - Employee Retention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
!pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Select relevant features for retention analysis
features = ['Overtime_Hours', 'Employee_Satisfaction_Score', 'Promotions', 'Training_Hours', 'Team_Size']
X = employee[features]
y = employee['Resigned'].astype(int)  

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X_train)
X1_test_scaled = scaler.transform(X_test)
```

    Requirement already satisfied: xgboost in /opt/anaconda3/lib/python3.12/site-packages (2.1.1)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (from xgboost) (1.26.4)
    Requirement already satisfied: scipy in /opt/anaconda3/lib/python3.12/site-packages (from xgboost) (1.13.1)



```python
#1. Logistic Regression Model 

# Initialize and fit the logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X1_train_scaled, y_train)

# Predictions
y1_pred_log_reg = log_reg.predict(X1_test_scaled)

# Evaluation
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y1_pred_log_reg)}")
print(f"Precision: {precision_score(y_test, y1_pred_log_reg)}")
print(f"Recall: {recall_score(y_test, y1_pred_log_reg)}")
print(f"F1 Score: {f1_score(y_test, y1_pred_log_reg)}")
print(classification_report(y_test, y1_pred_log_reg))
```

    Logistic Regression:
    Accuracy: 0.8989666666666667
    Precision: 0.0
    Recall: 0.0
    F1 Score: 0.0
                  precision    recall  f1-score   support
    
               0       0.90      1.00      0.95     26969
               1       0.00      0.00      0.00      3031
    
        accuracy                           0.90     30000
       macro avg       0.45      0.50      0.47     30000
    weighted avg       0.81      0.90      0.85     30000
    


    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /opt/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))



```python
#2. Random Forest Classifier Model

# Initialize and fit the random forest model
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X1_train_scaled, y_train)

# Predictions
y1_pred_rf = rf_clf.predict(X1_test_scaled)

# Evaluation
print("\nRandom Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y1_pred_rf)}")
print(f"Precision: {precision_score(y_test, y1_pred_rf)}")
print(f"Recall: {recall_score(y_test, y1_pred_rf)}")
print(f"F1 Score: {f1_score(y_test, y1_pred_rf)}")
print(classification_report(y_test, y1_pred_rf))
```

    
    Random Forest Classifier:
    Accuracy: 0.8975666666666666
    Precision: 0.09615384615384616
    Recall: 0.001649620587264929
    F1 Score: 0.003243593902043464
                  precision    recall  f1-score   support
    
               0       0.90      1.00      0.95     26969
               1       0.10      0.00      0.00      3031
    
        accuracy                           0.90     30000
       macro avg       0.50      0.50      0.47     30000
    weighted avg       0.82      0.90      0.85     30000
    



```python
#3. Gradient Boosting Model

# Initialize and fit the XGBoost model
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(X1_train_scaled, y_train)

# Predictions
y1_pred_xgb = xgb_clf.predict(X1_test_scaled)

# Evaluation
print("\nXGBoost Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y1_pred_xgb)}")
print(f"Precision: {precision_score(y_test, y1_pred_xgb)}")
print(f"Recall: {recall_score(y_test, y1_pred_xgb)}")
print(f"F1 Score: {f1_score(y_test, y1_pred_xgb)}")
print(classification_report(y_test, y1_pred_xgb))
```

    
    XGBoost Classifier:
    Accuracy: 0.8988333333333334
    Precision: 0.0
    Recall: 0.0
    F1 Score: 0.0
                  precision    recall  f1-score   support
    
               0       0.90      1.00      0.95     26969
               1       0.00      0.00      0.00      3031
    
        accuracy                           0.90     30000
       macro avg       0.45      0.50      0.47     30000
    weighted avg       0.81      0.90      0.85     30000
    



```python
# Model Comparision
print(f"Logistic Regression accuracy: {'accuracy'} = 0.89897")
print(f"Random Forest Classifier: {'accuracy'} = 0.89757")
print(f"Gradient Boost Classifier: {'accuracy'} = 0.89888")

print(f"Logistic Regression accuracy: {'precision'} = 0.0")
print(f"Random Forest Classifier: {'accuracy'} = 0.1")
print(f"Gradient Boost Classifier: {'accuracy'} = 0.0")

print(f"Logistic Regression accuracy: {'Recall'} = 0.0")
print(f"Random Forest Classifier: {'Recall'} = 0.02")
print(f"Gradient Boost Classifier: {'Recall'} = 0.0")

print(f"Logistic Regression accuracy: {'F1_score'} = 0.0")
print(f"Random Forest Classifier: {'F1_score'} = 0.032")
print(f"Gradient Boost Classifier:{'F1_score'} = 0.0")
```

    Logistic Regression accuracy: accuracy = 0.89897
    Random Forest Classifier: accuracy = 0.89757
    Gradient Boost Classifier: accuracy = 0.89888
    Logistic Regression accuracy: precision = 0.0
    Random Forest Classifier: accuracy = 0.1
    Gradient Boost Classifier: accuracy = 0.0
    Logistic Regression accuracy: Recall = 0.0
    Random Forest Classifier: Recall = 0.02
    Gradient Boost Classifier: Recall = 0.0
    Logistic Regression accuracy: F1_score = 0.0
    Random Forest Classifier: F1_score = 0.032
    Gradient Boost Classifier:F1_score = 0.0



```python
# Modelling - Satisfaction Insights 
# Select relevant features for satisfaction analysis
features = ['Work_Hours_Per_Week', 'Team_Size', 'Training_Hours']
X = employee[features]
y = employee['Employee_Satisfaction_Score']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X2_train_scaled = scaler.fit_transform(X_train)
X2_test_scaled = scaler.transform(X_test)
```


```python
# 1. Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
y2_pred_dt = decision_tree.predict(X_test)

print("\nDecision Tree Regressor:")
print(f"R^2: {r2_score(y_test, y2_pred_dt)}")
print(f"MSE: {mean_squared_error(y_test, y2_pred_dt)}")
```

    
    Decision Tree Regressor:
    R^2: -0.7625770923335955
    MSE: 2.3299300332888886



```python
from sklearn.ensemble import RandomForestRegressor

#2. Random Forest Regressor
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)
y2_pred_rf = random_forest.predict(X_test)

print("\nRandom Forest Regressor:")
print(f"R^2: {r2_score(y_test, y2_pred_rf)}")
print(f"MSE: {mean_squared_error(y_test, y2_pred_rf)}")
```

    
    Random Forest Regressor:
    R^2: -0.32004165750671043
    MSE: 1.7449476203876721



```python
from sklearn.svm import SVR

# 3. Support Vector Regression (SVR)
svr = SVR()
svr.fit(X2_train_scaled, y_train)
y2_pred_svr = svr.predict(X2_test_scaled)

print("\nSupport Vector Regression:")
print(f"R^2: {r2_score(y_test, y2_pred_svr)}")
print(f"MSE: {mean_squared_error(y_test, y2_pred_svr)}")
```

    
    Support Vector Regression:
    R^2: -0.0027881095962642988
    MSE: 1.325573867796031



```python
#Model Comparison 
print(f"Decission Tree Regressor: {'R^2'} = -0.762")
print(f"Random Forest Classifier: {'R^2'} = -0.320")
print(f"Supprot Vector Regressor: {'R^2'} = -0.002")

print(f"Decission Tree Regressor: {'MSE'} = 2.33")
print(f"Random Forest Classifier: {'MSE'} = 1.74")
print(f"Supprot Vector Regressor: {'MSE'} = 1.32")
```

    Decission Tree Regressor: R^2 = -0.762
    Random Forest Classifier: R^2 = -0.320
    Supprot Vector Regressor: R^2 = -0.002
    Decission Tree Regressor: MSE = 2.33
    Random Forest Classifier: MSE = 1.74
    Supprot Vector Regressor: MSE = 1.32

