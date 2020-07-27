import datetime
import logging
import os

import numpy as np
import pandas as pd

from utils.DatabaseHelper import get_auxiliary_data, get_distinct_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AuxiliaryDataFrameProcessor.py')

def log_event(message):
    current_time = datetime.datetime.now()
    logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

dataframe_columns = ['item_id', 'tour_id', 'name', 'from_date', 'to_date', 'accommodation_type', 'id_type', 'destination_country', 'destination_list',
           'nightly_price', 'avg_nightly_price', 'min_price', 'discount', 'duration', 'info', 'valid_from', 'valid_to']

def dictionary_creator(list_to_map):
    """
        Method is responsible of creating a dictionary for the values returned from database
    :param list_to_map: map to create dictionary
    :return: dictionary
    """
    item_dict = {}
    counter = 0
    for item in list_to_map:
        item_dict[counter] = item
        counter += 1

    return item_dict

def encode_discount_prices(row_value, is_cash, respective_dict):
    """
        Encode the discount information based on the discount type and value.
    :param row_value: Dataframe row
    :param is_cash: Is discount cash? If not, it is percentage
    :param respective_dict: Discount Dictionary to use for the encoding
    :return: OHE Vector
    """
    vector = np.zeros(len(respective_dict), dtype='int8')

    if is_cash:
        if row_value <= 5:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(5)]
        elif 5 < row_value <= 10:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(10)]
        elif 10 < row_value <= 15:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(15)]
        elif 15 < row_value <= 20:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(20)]
        elif 20 < row_value:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(25)]
    else:
        if row_value <= 500:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(500)]
        elif 500 < row_value <= 1500:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(1500)]
        elif 1500 < row_value <= 2000:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(2000)]
        elif 2000 < row_value <= 3000:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(3000)]
        elif 3000 < row_value:
            encoding = list(respective_dict.keys())[list(respective_dict.values()).index(12000)]

    vector[encoding] = 1
    return vector

def encode_tour_prices(nightly_price, avg_nightly_price, min_price):
    """
        Method responsible for encoding the price values.
        After we investigate the prices, we divided the price ranges to 5 segment. Then, we introduce buffer beteen each segment to fit
        values better. One possible future work here to use Fuzzy Logic to decide the price interval
    :param nightly_price: The nightly price of the tour
    :param avg_nightly_price: Avg nightly price of the tour
    :param min_price: Minimum prices of the tour
    :return: OHE Vectors: vector_nightly_price, vector_avg_nightly_price, vector_min_price
    """
    vector_nightly_price = np.zeros(5, dtype='int8')
    vector_avg_nightly_price = np.zeros(5, dtype='int8')
    vector_min_price = np.zeros(5, dtype='int8')

    # Find the type of nightly price
    if nightly_price <= nightly_price_slice_one + nightly_price_buffer:
        vector_nightly_price[0] = 1
    elif nightly_price_slice_one < nightly_price <= nightly_price_slice_two + nightly_price_buffer:
        vector_nightly_price[1] = 1
    elif nightly_price_slice_two < nightly_price <= nightly_price_slice_three + nightly_price_buffer:
        vector_nightly_price[2] = 1
    elif nightly_price_slice_three < nightly_price <= nightly_price_slice_four + nightly_price_buffer:
        vector_nightly_price[3] = 1
    else:
        vector_nightly_price[4] = 1

    # Find the type of avg_nightly_price
    if avg_nightly_price <= avg_nightly_price_slice_one - avg_nightly_price_buffer:
        vector_avg_nightly_price[0] = 1
    elif avg_nightly_price_slice_one < avg_nightly_price <= avg_nightly_price_slice_two + avg_nightly_price_buffer:
        vector_avg_nightly_price[1] = 1
    elif avg_nightly_price_slice_two < avg_nightly_price <= avg_nightly_price_slice_three + avg_nightly_price_buffer:
        vector_avg_nightly_price[2] = 1
    elif avg_nightly_price_slice_three < avg_nightly_price <= avg_nightly_price_slice_four + avg_nightly_price_buffer:
        vector_avg_nightly_price[3] = 1
    else:
        vector_avg_nightly_price[4] = 1

    # Find the type of min_price
    if min_price <= min_price_slice_one - min_price_buffer:
        vector_min_price[0] = 1
    elif min_price_slice_one < min_price <= min_price_slice_two + min_price_buffer:
        vector_min_price[1] = 1
    elif min_price_slice_two < min_price <= min_price_slice_three + min_price_buffer:
        vector_min_price[2] = 1
    elif min_price_slice_three < min_price <= min_price_slice_four + min_price_buffer:
        vector_min_price[3] = 1
    else:
        vector_min_price[4] = 1

    return vector_nightly_price, vector_avg_nightly_price, vector_min_price

def one_hot_encode_row_value(row_value, respective_dict):
    """
        Method is responsible of One-Hot-Encode the given value based on the respective dictionary.
    :param row_value: Dataframe row value
    :param respective_dict: Dictionary to use during OHE
    :return: One-Hot-Encoded Vector
    """
    vector = np.zeros(len(respective_dict), dtype='int8')
    encoding = list(respective_dict.keys())[list(respective_dict.values()).index(row_value)]
    vector[encoding] = 1
    return vector

def one_hot_encode_destination(country, destination_list, respective_dict):
    """
        Method encodes the destination country and follow up countries (if exist) into OHE form.
    :param country: Destination country
    :param destination_list: Follow up destination
    :param respective_dict: Destination dictionary
    :return: OHE Vector
    """
    vector = np.zeros(len(respective_dict), dtype='int8')

    new_countries = [] if len(country) == 0 else country.split(':')
    for c in new_countries:
        encoding = list(respective_dict.keys())[list(respective_dict.values()).index(c.lower())]
        vector[encoding] = 1

    new_destinations = [] if len(destination_list) == 0 else destination_list.split(':')
    for c in new_destinations:
        encoding = list(respective_dict.keys())[list(respective_dict.values()).index(c.lower())]
        vector[encoding] = 1

    return vector

def process_tour_data(row):
    """
        Method will iterate all the dataframe row and perform encodings for the following columns:
            tour_month, discount, encoded_accommodation_type, encoded_destination, encoded_duration, encoded_id_type
            nightly_price, avg_nightly_price, min_price
    :param row: Dataframe row
    :return: Encoded column values
    """
    # OHE Avg Nightly Price, OHE Nightly Price, OHE Min Price
    encoded_prices = encode_tour_prices(nightly_price=row.nightly_price,
                                        avg_nightly_price=row.avg_nightly_price,
                                        min_price=row.min_price)

    # OHE Month
    if row.from_date is not None:
        encoded_tour_month = one_hot_encode_row_value(row_value=row.from_date.month, respective_dict=dict_months)
    else:
        encoded_tour_month = one_hot_encode_row_value(row_value=row.to_date.month, respective_dict=dict_months)

    if row.discount <= 100:
        encoded_discount_type = encode_discount_prices(row_value=row.discount, is_cash=True, respective_dict=dict_discounts_percent)
    else:
        encoded_discount_type = encode_discount_prices(row_value=row.discount, is_cash=False, respective_dict=dict_discounts_cash)

    # OHE Item Category
    encoded_accommodation_type = one_hot_encode_row_value(row_value=int(row.accommodation_type), respective_dict=dict_accommodation)

    # OHE Destination
    encoded_destination = one_hot_encode_destination(country=row.destination_country, destination_list=row.destination_list, respective_dict=dict_destinations)

    # OHE Duration
    encoded_duration = one_hot_encode_row_value(row_value=row.duration, respective_dict=dict_duration)

    # OHE ID Type
    encoded_id_type = one_hot_encode_row_value(row_value=row.id_type, respective_dict=dict_id_type)

    row.nightly_price = encoded_prices[0]
    row.avg_nightly_price = encoded_prices[1]
    row.min_price = encoded_prices[2]

    row.from_date = encoded_tour_month
    row.discount = encoded_discount_type
    row.accommodation_type = encoded_accommodation_type
    row.destination_country = encoded_destination
    row.duration = encoded_duration
    row.id_type = encoded_id_type

    return row

def merge_all_vectors(row):
    """
        Method will concatenate all decided columns into one giant vector to represent content metadata.
    :param row: Dataframe Rows
    :return: One big vector
    """
    return np.concatenate((row.from_date, row.accommodation_type, row.id_type, row.destination_country, row.nightly_price, row.avg_nightly_price, row.min_price, row.discount, row.duration))

# Query auxiliary data from the database and insert it into a dataframe
auxiliary_data = get_auxiliary_data()
auxiliary_dataframe = pd.DataFrame(data=auxiliary_data, columns=dataframe_columns)

# Investigate the price information for the tours. We want to divide tour prices to 5 different segments
# by the respective quantile values. Also, create a buffer for the prices close to upper segment
avg_nightly_price_slice_one = auxiliary_dataframe['avg_nightly_price'].quantile(.10)
avg_nightly_price_slice_two = auxiliary_dataframe['avg_nightly_price'].quantile(.25)
avg_nightly_price_slice_three = auxiliary_dataframe['avg_nightly_price'].quantile(.50)
avg_nightly_price_slice_four = auxiliary_dataframe['avg_nightly_price'].quantile(.75)
avg_nightly_price_slice_five = auxiliary_dataframe['avg_nightly_price'].quantile(.95)
avg_nightly_price_buffer = 500.

nightly_price_slice_one = auxiliary_dataframe['nightly_price'].quantile(.10)
nightly_price_slice_two = auxiliary_dataframe['nightly_price'].quantile(.25)
nightly_price_slice_three = auxiliary_dataframe['nightly_price'].quantile(.50)
nightly_price_slice_four = auxiliary_dataframe['nightly_price'].quantile(.75)
nightly_price_slice_five = auxiliary_dataframe['nightly_price'].quantile(.95)
nightly_price_buffer = 1000.

min_price_slice_one = auxiliary_dataframe['min_price'].quantile(.10)
min_price_slice_two = auxiliary_dataframe['min_price'].quantile(.25)
min_price_slice_three = auxiliary_dataframe['min_price'].quantile(.55)
min_price_slice_four = auxiliary_dataframe['min_price'].quantile(.75)
min_price_slice_five = auxiliary_dataframe['min_price'].quantile(.90)
min_price_buffer = 1000.


dest_country, discounts_percent, discounts_cash, accommodation, id_type, duration = get_distinct_info()

dict_discounts_percent = dictionary_creator(list_to_map=discounts_percent)
dict_discounts_cash = dictionary_creator(list_to_map=discounts_cash)
dict_destinations = dictionary_creator(list_to_map=dest_country)
dict_accommodation = dictionary_creator(list_to_map=accommodation)
dict_id_type = dictionary_creator(list_to_map=id_type)
dict_duration = dictionary_creator(list_to_map=duration)
dict_months = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12}

# After investigating the discount types, we decided to divide them 5 different sections manually.
dict_discounts_percent = {0: 5, 1:10, 2:15, 3:20, 4:25}
dict_discounts_cash = {0: 500, 1:1500, 2:2000, 3:3000, 4:12000}

if not os.path.isfile('../source_data/auxiliary_dataframe.pickle'):
    log_event('| Start Processing the content metadata..')
    auxiliary_dataframe = auxiliary_dataframe.apply(process_tour_data, axis=1)
    log_event('| Processing is done..')

    auxiliary_dataframe.drop(columns=['name', 'info', 'destination_list'], inplace=True)
    auxiliary_dataframe.to_pickle("../source_data/auxiliary_dataframe_raw.pickle")
    log_event('| Raw Dataframe is saved to project file..')

    auxiliary_dataframe['all_vectors'] = auxiliary_dataframe.apply(merge_all_vectors, axis=1)
    auxiliary_dataframe.drop(columns=['accommodation_type', 'id_type', 'destination_country', 'nightly_price', 'avg_nightly_price', 'min_price', 'discount', 'duration'], inplace=True)
    auxiliary_df = auxiliary_dataframe[['item_id', 'tour_id', 'from_date', 'to_date', 'valid_from', 'valid_to', 'all_vectors']]
    auxiliary_df.to_pickle(f'../source_data/auxiliary_dataframe.pickle')
    log_event('| Processed Dataframe is save to project file..')
