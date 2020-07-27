import mysql.connector
from mysql.connector import errorcode

# Replace the below configuration based on your settings.
config = {
    'user': 'root',
    'password': 'password',
    'host': '127.0.0.1',
    'database': 'travel_agency',
    'raise_on_warnings': True,
    'auth_plugin': 'mysql_native_password'
}


def create_connection():
    """
    Method is used to establish connection with the database.
    :return: DB Connection Object
    """
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

def insert_evaluation(user_id, session_id, precision, recall, mrr, ndcg, predictor_name, trivial_prediction, catalog_prediction,
                      catalog_count, ground_truth, sequence, input_sequence, input_sequence_length, user_sequence_length, predictions):
    """
           Insert the evaluation result to the model_evaluation table.
    :param user_id: The ID of the user
    :param session_id: The ID of the user session
    :param precision: Precision Score
    :param recall: Recall Score
    :param mrr: MRR Score
    :param ndcg: nDCG Score
    :param predictor_name: Name of the recommender which produce the recommendations
    :param trivial_prediction: Is it a trivial prediction?
    :param catalog_prediction: Is the recommendation based on only catalog data?
    :param catalog_count: Number of the catalog visit in the sequence
    :param ground_truth: The item we want to predict
    :param sequence: User sequence
    :param input_sequence: Input stack of the recommender
    :param input_sequence_length: Length of the input stack of the recommender
    :param user_sequence_length: Length of the user sequence
    :param predictions: Top-K predictions
    """
    query = """ INSERT INTO travel_agency.model_evaluation 
                       (user_id, session_id, travel_agency.model_evaluation.precision, recall, mrr, ndcg, predictor_name, trivial_prediction, 
                       catalog_prediction, catalog_count, ground_truth, sequence, input_sequence, input_sequence_length, user_sequence_length, predictions) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s); """

    conn = create_connection()
    cursor = conn.cursor()

    data = (user_id, session_id, precision, recall, mrr, ndcg, predictor_name, trivial_prediction, catalog_prediction, catalog_count, ground_truth, sequence, input_sequence,
            input_sequence_length, user_sequence_length, predictions)

    cursor.execute(query, data)
    conn.commit()
    cursor.close()

def insert_model_performance(model_name, split_number, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
    """
           Insert the model performance for each split training to the model_performance table
    :param model_name: Name of the recommender which produce the recommendations
    :param split_number: Which split is the performance belong to
    :param epoch: Number of epochs
    :param train_loss: Train loss
    :param val_loss: Validation Loss
    :param train_accuracy: Accuracy Loss (In the case of Regression task, it is not accuracy but mean_absolute_error. Since metrics are dynamic, no code change required.
    :param val_accuracy: Validation Accuracy
    """
    query = """ INSERT INTO travel_agency.model_performance
                       (model_name, split_number, epoch, train_loss, val_loss, train_accuracy, val_accuracy) 
                VALUES (%s, %s, %s, %s, %s, %s, %s); """

    conn = create_connection()
    cursor = conn.cursor()

    data = (model_name, split_number, epoch, train_loss, val_loss, train_accuracy, val_accuracy)
    cursor.execute(query, data)
    conn.commit()

    cursor.close()

def insert_model_data(model_type, training_type, description, hyper_parameters, meta_data):
    """
           Insert model related information to database to keep track of the trained models.
    :param model_type: Name of the model
    :param training_type: training_options used to train the model (e.g. 8_epoch_512_batch)
    :param description: Description of the model like its abilities, features and labels.
    :param hyper_parameters: Given parameters
    :param meta_data: Model metadata e.g. {"8_epoch_512_batch": {"epochs": 8, "k_split": 4, "batch_size": 512, "max_trials": 3, . . . }
    """
    query = """ INSERT INTO travel_agency.lstm_models
                       (model_type, training_type, description, hyper_parameters, meta_data) 
                VALUES (%s, %s, %s, %s, %s); """

    conn = create_connection()
    cursor = conn.cursor()

    data = (model_type, training_type, description, hyper_parameters, meta_data)
    cursor.execute(query, data)
    conn.commit()
    cursor.close()

def fetch_user_item_seq_info_with_date():
    """
        Query the userID, objectID, sessionID and startDatetime from the database. The method will get the date in ordered by userID and startDatetime
        so that we can get the every user visit in a sequential order
    :return: The list contains the queried data
    """
    query_user_ids = """ Select userID, objectID, sessionID, startDatetime from travel_agency.implicit_user_feedback order by userID, startDatetime; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, item_id, session_id, startDatetime) for user_id, item_id, session_id, startDatetime in data_tuple]
    return data

def fetch_all_items():
    """
        Query all the dedicated tour visits
    :return: The list contains the queried data
    """
    query_user_ids = """ SELECT DISTINCT (objectID) 
                         FROM travel_agency.implicit_user_feedback 
                         WHERE objectID <> 0 ORDER BY objectID;"""

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [item for t in data_tuple for item in t]
    return data

def get_all_catalog_items_less_columns():
    """
        Query catalog visits. This method is one of the methods to use to create item dictionary thats why we only need to query required columns.
    :return: The list contains the queried data
    """
    query_user_info = """ SELECT userID, objectID, sessionID, objectsListed
                         FROM travel_agency.implicit_user_feedback 
                         WHERE pageType = 'katalog' OR pageType = 'index' AND LENGTH(objectsListed) - LENGTH(REPLACE(objectsListed, ';', '')) > 1
                         ORDER BY userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_info)

    data_tuple = cursor.fetchall()
    data = [(userID, objectID, sessionID, objectsListed) for userID, objectID, sessionID, objectsListed in data_tuple]
    return data


def get_all_catalog_items():
    """
        Query catalog visits including all the columns
    :return: The list contains the queried data
    """
    query_user_info = """ SELECT userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, objectsListed, travel_agency.implicit_user_feedback.logFile
                         FROM travel_agency.implicit_user_feedback
                         WHERE pageType = 'katalog' OR pageType = 'index' OR pageType = 'informace'
                         ORDER BY userID; """
    # WHERE pageType = 'katalog' OR pageType = 'index' OR pageType = 'informace' AND LENGTH(objectsListed) - LENGTH(REPLACE(objectsListed, ';', '')) > 1
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_info)

    data_tuple = cursor.fetchall()
    data = [(userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, objectsListed, logFile) for
            userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, objectsListed, logFile in data_tuple]
    return data


def get_auxiliary_data():
    """
        Query for the content metadata
    :return: The list contains the queried data
    """
    query_data = """ SELECT id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from, valid_to
                     FROM travel_agency.content_base_tour_details
                     ORDER BY id_serial ASC, id_record DESC"""

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_data)

    data_tuple = cursor.fetchall()
    data = [(
            id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from,
            valid_to) for
            id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from, valid_to
            in data_tuple]
    return data


def split_country_name(c_name, c_list):
    """
        Method to process country names. It splits the given string and insert the destination country to the respective list.
        If the country is already in the list, we will not add it again.
    :param c_name: country name
    :param c_list: country list
    """
    if ':' in c_name:
        countries = c_name.split(':')
        for c in countries:
            if c not in c_list:
                c_list.append(c.lower())
    else:
        if len(c_name) > 0:
            if c_name not in c_list:
                c_list.append(c_name.lower())


def get_distinct_info():
    """
        Method queries the country, discount, accommodation, id_typ, length, follow_up_destination and destination columns and process them
    :return: The lists contains the processed data
    """
    query_data = """ SELECT DISTINCT zeme, sleva, ubytovani_kategorie, id_typ, delka, informace_list, destinace
                     FROM travel_agency.content_base_tour_details """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_data)

    data_tuple = cursor.fetchall()

    dest_country, percent_discount_types, cash_discount_types, accommodation_type, id_type, duration = [], [], [], [], [], []

    for i in data_tuple:

        country = i[0].lower()
        discount = int(i[1])
        accommodation = int(i[2])
        id_typ = i[3]
        length = i[4]
        follow_up_destination = i[5].lower()
        dest = i[6].lower()

        split_country_name(country, dest_country)
        split_country_name(follow_up_destination, dest_country)
        split_country_name(dest, dest_country)

        if discount > 100:  # Change based on the real value
            if discount not in cash_discount_types:
                cash_discount_types.append(discount)
        else:
            if discount not in percent_discount_types:
                percent_discount_types.append(discount)

        if accommodation not in accommodation_type:
            accommodation_type.append(accommodation)

        if id_typ not in id_type:
            id_type.append(id_typ)

        if length not in duration:
            duration.append(length)

    destinations = sorted(dest_country)
    percent_discount_types = sorted(percent_discount_types)
    cash_discount_types = sorted(cash_discount_types)
    accommodation_type = sorted(accommodation_type)
    id_type = sorted(id_type)
    duration = sorted(duration)

    return destinations, percent_discount_types, cash_discount_types, accommodation_type, id_type, duration
