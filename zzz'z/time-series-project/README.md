# Time Series Project

## About the Project

The goal is to recommend a product line fit for expansion to the VP of Product. Through thoughtful analysis, we would like to explore if there is a product category that might be particularly profitable and stands out in terms of sales volume. If so, we will attempt to forecast potential gains in profit.

## Project Description

> "Superstore's mission is to be the preferred supplier of workspace solutions; from home-office to corporate office, we aspire to be the leading expert in workplace solutions for everyone!
>- Rachel Robbins-Mayhill, CEO

In addition, Superstore aims to be efficient in it's efforts to produce profit for shareholders. For this reason, we will explore how best to enhance profits by expanding product lines in one of our three categories: Office Supplies, Furniture, and Technology. To do this, we will analyze previous profits to find a category ripe for expansion, then use Time Series Analysis and Data Science techniques to project profits based on expansion in that category.

## Project Goals

Answer the VP of Products' question:

- Which product line should we expand?

Maximize effectiveness of product line expansion by identifying the product line with:

1. the highest profitability, and
1. the largest amount of room for growth in terms of sales volume.

## Initial Questions

- Is there a product category that has significantly higher profit margins than the others?

- Is there a product category that has significantly lower sales volume than the others?

## Data Dictionary

|  Variables             |    Definition                              |    DataType             |
| :--------------------:   | :----------------------------------------: | :--------------------: |
order_date (index)    |  Date order was placed                          |  datetime64[ns]    |
order_id              |  Order identifier assigned to each product name for each order | object |
ship_date             |  Date order was shipped                         | datetime64[ns]       |
ship_mode             |  Mode of shipping for delivery:  'Standard Class', 'First Class', 'Second Class', 'Same Day'  | object      |
segment               |  Customer type: Consumer, Cooperate, Home Office  |  object     |
country               |  Country to which shipment was delivered: 'United States'  |    object   |
city                  |  City to which shipment was delivered  |  object    |
state                 |  State to which shipment was delivered  |  object    |
postal_code           |  Postal code to which shipment was delivered   |  float64   |
sales                 |  Sale total for product id * quantity in given order ($USD)  | float64     |
quantity              |  Total number of specified product ordered  | float64     |
discount              |  Percentage of discount applied to order in decimal form  | float64     |
profit                |  Sales - Product Cost  | float64     |
category              |  Category the product belongs to  | object    |
sub-category          |  Subcategory the product belongs to  |  object    |
customer_name         |  Name of customer   | object     |
product_name          |  Name of product  |  object    |
region_name           |  General area of US where order was placed: 'Central', 'South', 'East', 'West'  |  object    |
profit_per_item *     |  Profit margin per item  |  float64    |
| * | feature engineered | |

## Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the Superstore database. The env.py should also contain a function named get_db_url() that establishes the string value of the database url. Store that env file locally in the repository.
2. clone my repo (including the acquire.py, prepare.py, explore.py, and model.py modules) (confirm .gitignore is hiding your env.py file)
3. libraries used are pandas, matplotlib, seaborn, numpy, sklearn, math.

## The Plan

Method:

### 1. Imports

- Imports used can be found in `imports.py`. (Please ensure libraries are installed for package support).

### 2. Acquisition

- In this stage, we obtained Superstore customer data by querying the Codeup MySQL database hosted at data.codeup.com. The original source of this data was db.codeup.com.

### 3. Preparation

- We cleaned and prepped the data by:
    - removing all observations that included null values
    - renaming columns for readability
    - changing data types where appropriate
    - adding a feature: profit_per_item
    - adding a feature: sales_per_item
    - set the index to `datetime`

### 4. Exploration

- We conducted an initial exploration of the data by examing relationships between each of the features and treated profit as a target
- Next, we explored further using the keen eyes of a weathered data scientist and premier tools such as Pandas, Python, Statsmodels, etc..., to answer the initial questions posed above.
- Findings:
    - The Technology product category has higher profit margins than either Office Supplies or Furniture, yet has a much lower total sales volume. We expect that this means there is both high profit potential and room for growth in this category. 

### 5. Forecasting

- We used data from 2014 - 2016 to determine which product category has the most potential for expansion, then modeled what might have happened if we expanded this category in 2017, and compared those projected profits to the actual 2017 profits. 

## How did we do?

If we had increased technology sales as suggested for the year 2017, we could have expected an average **monthly increase in profits of 63%**, yielding a **96% increase in total profit for the year**. 

Therefore, we strongly recommend conducting a similar analysis to determine which product line to expand in future years.

## Key Findings

We determined that the technology market has the best oppurtunity for expansion due to:

- highest profits per item
- lowest sales volume

## Recommendations

Profits on technology sales are higher than other markets. Additionally, sales volume is among the lowest categories. Based on profitability and projected room for growth, expansion efforts should be targeted on the technology sector.

## Next Steps

Given more time, we would like to:

- explore regional effects on profits
- explore regional effects on profits per market category
- explore office supplies as a secondary contender for expansion
- use time-series-analysis to predict 2018 profits, both with and without implementing our recommendations
- work on  a more nuanced simulation for 2014-2016

