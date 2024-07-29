# codetech-task-1
Name:M LEELAKRISHNA

Company:CODETECH IT SOLUTIONS

ID:CT6DS708


Date:June25th to August10th,2024

Mentor:Muzammil Ahmed

# Overview of the Project

# Project:Explatory Data Analsis (EDA) on Car Data Analysis  Dataset

# Objective
The primary objective of this project is to perform an exploratory data analysis (EDA) on a dataset of used cars. The analysis aims to uncover insights into the distribution and relationships within the data, which can be useful for further predictive modeling or decision-making processes.

# Key Activities
## 1.Loading and Inspecting Data:

- Importing the dataset.
- Displaying the first and last few records.
- Checking data types and null values.
- Calculating the percentage of missing values.
## 2.Data Cleaning:

- Removing unnecessary columns.
- Creating new columns, such as Car_Age.
- Splitting the Name column into Brand and Model.
- Standardizing inconsistent brand names.
## 3.Summary Statistics:

- Generating summary statistics for numerical and categorical columns.
## 4.Identifying Categorical and Numerical Variables:

- Identifying columns for further analysis.
## 5.Visualizing Numerical Columns:

- Plotting histograms and box plots to understand distribution and identify outliers.
## 6.Visualizing Categorical Columns:

- Creating bar plots to visualize the distribution of categorical variables.
## 7.Log Transformation:

- Applying log transformation to Kilometers_Driven and Price to normalize distributions.
## 8.Visualizing Transformed Data:

- Plotting the distribution of log-transformed columns.
##  9.Pair Plot:

- Generating a pair plot to examine relationships between numerical variables.
## 10.Bar Plots of Grouped Means:

- Creating bar plots showing mean log-transformed Price grouped by various categorical variables.
## 11.Line Plot:

- Plotting Kilometers_Driven and Price on the same graph to compare trends.
## 12.Scatter Plot:

- Creating a scatter plot to visualize the relationship between Kilometers_Driven and Price.
# Technology Used
- Pandas: For data manipulation and analysis.
- Numpy: For numerical operations.
- Matplotlib: For creating static, animated, and interactive visualizations.
- Seaborn: For statistical data visualization.

# Used Cars Data Analysis

This project performs an extensive exploratory data analysis (EDA) on a dataset of used cars. The steps include data loading, cleaning, transformation, and visualization.

## Key Steps

1. **Loading and Inspecting Data:**
    - Imported the dataset using Pandas.
    - Displayed the first and last few records.
    - Checked for data types and null values.
    - Calculated the percentage of missing values.

2. **Data Cleaning:**
    - Removed the `S.No.` column.
    - Created a new column `Car_Age` to represent the age of the car.
    - Split the `Name` column into `Brand` and `Model`.
    - Standardized inconsistent brand names.

3. **Summary Statistics:**
    - Generated summary statistics for numerical and categorical columns.

4. **Categorical and Numerical Variables:**
    - Identified categorical and numerical columns for further analysis.

5. **Visualizing Numerical Columns:**
    - Plotted histograms and box plots to understand the distribution and identify outliers.

6. **Visualizing Categorical Columns:**
    - Created bar plots to visualize the distribution of categorical variables.

7. **Log Transformation:**
    - Applied log transformation to `Kilometers_Driven` and `Price` to normalize their distributions.

8. **Visualizing Transformed Data:**
    - Plotted the distribution of log-transformed columns.

9. **Pair Plot:**
    - Generated a pair plot to examine relationships between numerical variables.

10. **Bar Plots of Grouped Means:**
    - Created bar plots showing mean log-transformed `Price` grouped by various categorical variables such as `Location`, `Transmission`, `Fuel_Type`, `Owner_Type`, `Brand`, `Model`, `Seats`, and `Car_Age`.

11. **Line Plot:**
    - Plotted `Kilometers_Driven` and `Price` on the same graph to compare trends.

12. **Scatter Plot:**
    - Created a scatter plot to visualize the relationship between `Kilometers_Driven` and `Price`.

## Libraries Used

- Pandas
- Numpy
- Matplotlib
- Seaborn

## Visualizations

The project includes various visualizations such as histograms, box plots, bar plots, line plots, scatter plots, and pair plots to help understand the data distribution and relationships.

## How to Run

1. Ensure all the necessary libraries are installed:
    ```sh
    pip install pandas numpy matplotlib seaborn
    ```

2. Place the `used_cars_data.csv` file in the working directory.

3. Run the script to perform data analysis and generate visualizations.

## Project Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv("used_cars_data.csv")

# Initial data inspection
data.head()
data.tail()
data.info()
data.nunique()
data.isnull().sum()
(data.isnull().sum()/(len(data)))*100

# Data cleaning
data = data.drop(['S.No.'], axis=1)
from datetime import date
data['Car_Age'] = date.today().year - data['Year']
data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper", "Land": "Land Rover"}, inplace=True)

# Summary statistics
data.describe().T
data.describe(include='all').T

# Identify categorical and numerical columns
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:", cat_cols)
print("Numerical Variables:", num_cols)

# Visualize numerical columns
for col in num_cols:
    print(col)
    print('Skew :', round(data[col].skew(), 2))
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.show()

# Visualize categorical columns
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')
sns.countplot(ax=axes[0, 0], x='Fuel_Type', data=data, color='blue', order=data['Fuel_Type'].value_counts().index)
sns.countplot(ax=axes[0, 1], x='Transmission', data=data, color='blue', order=data['Transmission'].value_counts().index)
sns.countplot(ax=axes[1, 0], x='Owner_Type', data=data, color='blue', order=data['Owner_Type'].value_counts().index)
sns.countplot(ax=axes[1, 1], x='Location', data=data, color='blue', order=data['Location'].value_counts().index)
sns.countplot(ax=axes[2, 0], x='Brand', data=data, color='blue', order=data['Brand'].head(20).value_counts().index)
sns.countplot(ax=axes[2, 1], x='Model', data=data, color='blue', order=data['Model'].head(20).value_counts().index)
axes[1][1].tick_params(labelrotation=45)
axes[2][0].tick_params(labelrotation=90)
axes[2][1].tick_params(labelrotation=90)

# Log transformation
def log_transform(data, col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname] + 1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    data.info()

log_transform(data, ['Kilometers_Driven', 'Price'])

# Visualizing transformed data
sns.distplot(data["Kilometers_Driven_log"], axlabel="Kilometers_Driven_log")
plt.show()

# Pair plot
plt.figure(figsize=(13, 17))
sns.pairplot(data=data.drop(['Kilometers_Driven', 'Price'], axis=1))
plt.show()

# Bar plots of grouped means
fig, axarr = plt.subplots(4, 2, figsize=(12, 18))
data.groupby('Location')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)
axarr[0][0].set_title("Location Vs Price", fontsize=18)
data.groupby('Transmission')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
axarr[0][1].set_title("Transmission Vs Price", fontsize=18)
data.groupby('Fuel_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][0], fontsize=12)
axarr[1][0].set_title("Fuel_Type Vs Price", fontsize=18)
data.groupby('Owner_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12)
axarr[1][1].set_title("Owner_Type Vs Price", fontsize=18)
data.groupby('Brand')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][0], fontsize=12)
axarr[2][0].set_title("Brand Vs Price", fontsize=18)
data.groupby('Model')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12)
axarr[2][1].set_title("Model Vs Price", fontsize=18)
data.groupby('Seats')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][0], fontsize=12)
axarr[3][0].set_title("Seats Vs Price", fontsize=18)
data.groupby('Car_Age')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][1], fontsize=12)
axarr[3][1].set_title("Car_Age Vs Price", fontsize=18)
plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()

# Line plot
plt.figure(figsize=(10, 5))
plt.plot(data['Kilometers_Driven'], label='Kilometers_Driven')
plt.plot(data['Price'], label='Price')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(data['Kilometers_Driven'], data['Price'], color='blue', marker='o')
plt.show()
