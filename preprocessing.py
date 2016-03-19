import numpy as np
import pandas as pd
from sklearn import preprocessing

def split_by_days(data_days):
  days = np.zeros((len(data_days), 3))
  i = 0
  for day in data_days:
    if 1 <= day < 10:
      days[i, 0] = 1
    elif 10 <= day < 20:
      days[i, 1] = 1
    else:
      days[i, 2] = 1
    i += 1
  return pd.DataFrame(days, columns=['FROM_1_TO_10', 'FROM_10_TO_20', 'FROM_20_TO_31'])

def split_by_years(data_years):
  years = np.zeros((len(data_years), 5))
  i = 0
  for year in data_years:
    if 1990 <= year < 1995:
      years[i, 0] = 1
    elif 1995 <= year < 2000:
      years[i, 1] = 1
    elif 2000 <= year < 2005:
      years[i, 2] = 1
    elif 2005 <= year < 2010:
      years[i, 3] = 1
    else:
      years[i, 4] = 1
    i += 1
  return  pd.DataFrame(years, columns=['1990_TO_1995', '1995_TO_2000', '2000_TO_2005', '2005_TO_2010', '2010_TO_2015'])

def process_data(data):
  print("data shape before processing: (" + str(data.shape[0]) + "," + str(data.shape[1]) + ")")
  encoder = preprocessing.LabelEncoder()
  special_parts = pd.DataFrame(encoder.fit_transform(data['SPECIAL_PART']).T, columns=['SPECIAL_PART'])

  # transaction_date
  transaction_years = pd.get_dummies(data['TRANSACTION_DATE'].dt.year)
  transaction_months = pd.get_dummies(data['TRANSACTION_DATE'].dt.month)
  transaction_days = split_by_days(data['TRANSACTION_DATE'].dt.day)

  # product_price
  product_price = pd.DataFrame(preprocessing.normalize(data['PRODUCT_PRICE']).T, columns=['PRODUCT_PRICE'])

  # gross_sales
  gross_sales = pd.DataFrame(preprocessing.normalize(data['GROSS_SALES']).T, columns=['GROSS_SALES'])

  # customer_segment1
  customer_segment1 = pd.get_dummies(data['CUSTOMER_SEGMENT1'])

  # customer_type1
  customer_type1 = pd.get_dummies(data['CUSTOMER_TYPE1'])

  # customer_account_type
  customer_account_type = pd.get_dummies(data['CUSTOMER_ACCOUNT_TYPE'])

  # customer_first_order_date
  first_order_years = split_by_years(data['CUSTOMER_FIRST_ORDER_DATE'].dt.year)
  first_order_months = pd.get_dummies(data['CUSTOMER_FIRST_ORDER_DATE'].dt.month)
  first_order_day = split_by_days(data['CUSTOMER_FIRST_ORDER_DATE'].dt.day)

  # brand
  brand = pd.get_dummies(data['BRAND'])

  # product_sales_unit
  product_sales_unit = pd.get_dummies(data['PRODUCT_SALES_UNIT'])

  # shipping_weigth
  shipping_weight = pd.DataFrame(preprocessing.normalize(data['SHIPPING_WEIGHT']).T, columns=['SHIPPING_WEIGHT'])

  # product_cost1
  product_cost1 = pd.DataFrame(preprocessing.normalize(data['PRODUCT_COST1']).T, columns=['PRODUCT_COST1'])

  # product_unit_of_measure
  product_unit_of_measure = pd.get_dummies(data['PRODUCT_UNIT_OF_MEASURE'])

  #order_source
  order_source = pd.get_dummies(data['ORDER_SOURCE'])

  # price_method
  price_method = pd.get_dummies(data['PRICE_METHOD'])

  temp = pd.concat([transaction_years, transaction_months, transaction_days, product_price, gross_sales,
                    customer_segment1, customer_type1, customer_account_type, first_order_years, first_order_months,
                    first_order_day, brand, product_sales_unit, shipping_weight, product_cost1, product_unit_of_measure,
                    order_source, price_method, special_parts], axis=1)

  for i in ['TRANSACTION_DATE', 'SPECIAL_PART', 'PRODUCT_PRICE', 'GROSS_SALES', 'CUSTOMER_SEGMENT1',
            'CUSTOMER_TYPE1', 'CUSTOMER_ACCOUNT_TYPE', 'CUSTOMER_FIRST_ORDER_DATE', 'BRAND', 'PRODUCT_SALES_UNIT',
            'SHIPPING_WEIGHT', 'PRODUCT_COST1', 'PRODUCT_UNIT_OF_MEASURE', 'ORDER_SOURCE', 'PRICE_METHOD']:
    data = data.drop(i, 1)

  data = pd.concat([data, temp], axis=1)
  print("data shape after processing: (" + str(data.shape[0]) + "," + str(data.shape[1]) + ")")

  data.to_csv("processed_data.csv", index=False)

data = pd.read_csv("data.csv", parse_dates=['TRANSACTION_DATE', 'CUSTOMER_FIRST_ORDER_DATE'])
process_data(data)