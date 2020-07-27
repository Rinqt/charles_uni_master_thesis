import pickle
from operator import itemgetter
import logging
import datetime

import pandas as pd
from utils import DatabaseHelper as DbHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainingDataProcessor.py')

def log_event(message):
    current_time = datetime.datetime.now()
    logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

def clean_objects_listed(row):
    """
    Method is responsible for pre-processing the objectsListed data coming from the database.

    objectsListed column structure is [objID: y_coordinate, x_coordinate, 1; objID2: y_coordinte, x_coordinate, 1; . . .]
    Note that initially the source data is wrong. Web page is logging the x_coordinate of the image to the place of y_coordinate and vise versa/

    Workflow:
        1. Check if columns has data. If don't simply insert "No Catalog Item" and return the dataframe row.
        2. Split the objects_listed column data by ";" to have one item data
        3. Iterate through every split in the objects_listed
            3.1 We do not need the last 1 and the comma before that so it can be removed.
            3.2 Split the remaining data by ':'
            3.3 Design the split data as [int(item_id), int(x_coordinate), int(y_coordinate)] ... ] and insert to a new list.

    @param row: The dataframe row
    @return: The updated objects_listed column for the given row
    """
    # objectsListed row will be ITEMID: Y_COORDINATE, X_COORDINATE, 1
    objects_listed = row.catalog_item_list

    if len(objects_listed) == 0:
        return 'No Catalog Item'

    # Split them by ';'
    objects_listed = objects_listed.split(';')[:-1]  # Last one is always empty
    index = 0
    cleaned_list = []  # It will have [ ... [int(item_id), int(x_coordinate), int(y_coordinate)] ... ]

    for pair in objects_listed:  # pair >> item_id:y_coor,x_coor,1
        if len(pair) > 9:
            objects_listed[index] = pair[:-2]

            data = objects_listed[index].split(':')  # data >> ['item_id', 'y_coor,x_coor']
            append = cleaned_list.append
            if len(data[0]) > 0:
                item_id = int(data[0])
                pixel_location = data[1].split(',')
                y_loc = int(pixel_location[0])
                x_loc = int(pixel_location[1])

                append([item_id, x_loc, y_loc])

                index += 1
        else:
            return 'No Catalog Item'

    cleaned_list = sorted(cleaned_list, key=itemgetter(2))
    return cleaned_list

def label_page_type(row):
    """
    Method is responsible for pre-processing the catalog item list. Note that this method is only called to increase human readability of
    the source dataframe. It will basically create a tuple contains [itemID, 'C'] which means that item is coming from a catalog page.

    Workflow:
        Label the catalog items with 'C' for better readability. e.g. (7046, 'C'), (7047, 'C')

    @param row: The dataframe row
    @return: The updated catalog item list
    """
    if isinstance(row.catalog_item_list, list):
        item_list = []
        append = item_list.append
        for item_info in row.catalog_item_list:
            append((item_info[0], 'C'))

        return item_list
    else:
        return 'No Catalog Item'

def clean_logs(row):
    """
    Method is responsible for pre-processing the logFile column of database.

    Workflow:
        1. First, check if user log contains any actions. If not, simply insert "No Log Found" and return the row value.
        2. Split the raw logs by newline. This will put each user action to a list
        3. Create a new list to insert cleaned actions. Then, iterate the user actions:
            |--> Replace the defined actions names 'MouseMove, to:', 'Scroll, to:', ' MouseClick' to 'MM', 'MS', and 'MC' respectively.
            |--> Then append actions to a new list in ['action_type', 'x_coordinate', 'y_coordinate'] structure.

    @param row: The dataframe row
    @return: The updated user_log_list column
    """
    if len(row.user_log_list) == 0:
        return 'No Log Found'

    # Get the text after the 2: since the first 2 is always \n
    log_file = row.user_log_list[2:]

    split_log = log_file.split('\n')

    log_list = []
    for action in split_log:

        if 'null' in action:
            continue

        temp_log = action.split(';')
        date = temp_log[0].lstrip()

        if len(temp_log) >= 2:
            moves = temp_log[1].split('  ')

            operation = ''
            index = 0
            cleaner_actions = []
            append = cleaner_actions.append
            for m in moves:
                if ' MouseMove, to:' in m:
                    coordinates = m.replace(' MouseMove, to:', '').lstrip()
                    coordinates = coordinates.split(',')
                    x_coordinate = int(float(coordinates[0]))
                    y_coordinate = int(float(coordinates[1]))
                    operation = 'MM'
                    #cleaner_actions.append((x_coordinate, y_coordinate))
                    append((x_coordinate, y_coordinate))

                elif ' Scroll, to:' in m:
                    coordinates = m.replace(' Scroll, to:', '').lstrip()
                    coordinates = coordinates.split(',')
                    x_coordinate = int(float(coordinates[0]))
                    y_coordinate = int(float(coordinates[1]))
                    operation = 'MS'
                    cleaner_actions.append((x_coordinate, y_coordinate))
                elif ' MouseClick' in m:
                    if len(temp_log) > 2:
                        item_id = temp_log[2].replace('on oid=', '').lstrip()
                        operation = 'MC'
                        append(int(item_id))
                elif 'to:' in m:
                    coordinates = m.replace('to:', '').lstrip()
                    coordinates = coordinates.split(',')
                    if len(coordinates) == 2:
                        x_coordinate = int(float(coordinates[0]))
                        y_coordinate = int(float(coordinates[1]))
                        append((x_coordinate, y_coordinate))
                index += 1

            log_list.append([date, operation, cleaner_actions])

    return log_list

def remove_last_catalog(row):
    """
    Method is responsible for removing the last catalog visit if the visit happend at the end of the user sequence. Once there is a catalog visit at the end of
    the user sequence, we need to remove it because the last item of the user sequence will be assigned as RNN Label.

    Workflow:
        1. Iterate through the visit actions and start removing the catalog visit at the end of the sequence.
        2. While removing the catalog, also remove the last element of catalog_item_list, user_log_list and good_catalog_items since they are given data
        for the last item of the sequence.
        Iterate until there is no catalog visit left at the end of the user sequence. If there is no visit left at the sequence, insert 'To Delete'
        string to the item_sequence column

    @param row: The dataframe row
    @return: The updated dataframe row
    """
    while row.item_sequence[-1] == 0:
        row.item_sequence = row.item_sequence[:-1]

        if row.catalog_item_list != -9.0:
            row.catalog_item_list = row.catalog_item_list[:-1]
            if len(row.catalog_item_list) == 0:
                row.catalog_item_list = 'No Item Found'

        if row.user_log_list != 'No Log Found':
            row.user_log_list = row.user_log_list[:-1]
            if len(row.user_log_list) == 0:
                row.user_log_list = 'No Log Found'

        if row.good_catalog_items != -9.0:
            row.good_catalog_items = row.good_catalog_items[:-1]
            if len(row.good_catalog_items) == 0:
                row.good_catalog_items = 'No Catalog Items'

        if len(row.item_sequence) == 0:
            row.item_sequence = 'To Delete'
            return row

    return row

def get_interacted_catalog_items(raw_df):
    """
    Method is responsible for finding the catalog items that user interacted with. There are three types of interaction:
        1. Item-Click: User click on the item.
        2. Mouse-Hover: User mouse were on the tour image.
        3. Not-Visible: Catalog item was not visible by the user

    Workflow:
        1. Simulate the user actions and find if user interacted with the catalog item:
        2. If yes; then label the interactions as tuple e.g. (itemID, 0) for the mouse hover action, (itemID, 1) for the item click action

    @param raw_df: Boolean to save the dataframe
    @return: Datafarme which contains 'good_catalog_items' column that represents the catalog items that user interacted during the session.
    """
    log_event('Method get_interacted_catalog_items() started..')

    user_catalog_items = raw_df['catalog_item_list'].values
    user_log = raw_df['user_log_list'].values
    user_window_x_list = raw_df['window_size_x'].values
    user_window_y_list = raw_df['window_size_y'].values
    page_x_list = raw_df['page_size_x'].values
    page_y_list = raw_df['page_size_y'].values

    # Create a new row to append catalog items that user hover her/his mouse
    raw_df['good_catalog_items'] = None

    default_image_size_x = 412
    default_image_size_y = 90

    cursor_x_pos = 0
    cursor_y_pos = 0

    counter = 0

    for catalog_list, log_list, user_window_x, user_window_y, page_x, page_y in zip(user_catalog_items, user_log, user_window_x_list, user_window_y_list, page_x_list, page_y_list):
        if catalog_list != 'No Catalog Item':
            buffer = 0
            user_window_y_start = 0
            mouse_was_here = []

            if log_list != 'No Log Found':
                for entry in log_list:
                    action_type = entry[1]
                    action = entry[2]

                    for pos in action:
                        if action_type == 'MC':
                            # If there is mouse click, pos will give the itemID
                            mouse_was_here.append((pos, 1))
                        else:
                            if action_type == 'MS':
                                y_moved = pos[1] - cursor_y_pos
                                cursor_y_pos += y_moved
                                user_window_y += y_moved
                                user_window_y_start += y_moved
                                buffer = default_image_size_y - int((pos[1] % default_image_size_y))
                            else:
                                # Update the cursor location
                                cursor_x_pos = pos[0]
                                cursor_y_pos = pos[1]

                            # Check if cursor is on an item:
                            for visible_item in catalog_list:
                                X_Pos = visible_item[1]
                                Y_Pos = visible_item[2]

                                # If the y_coordinate of the image is bigger than the y_coordinate of the mouse, we do not have to iterate all the catalog_list since items
                                # in the list are ordered by their y_coordinates.
                                if Y_Pos > cursor_y_pos:
                                    break

                                # Create the image borders
                                image_border_x_start = X_Pos
                                image_border_x_end = X_Pos + default_image_size_x
                                image_border_y_start = Y_Pos
                                image_border_y_end = Y_Pos + default_image_size_y

                                # If item is in the visible area of the user window:
                                if (image_border_x_start <= user_window_x) and ((user_window_y_start - buffer) <= image_border_y_start <= user_window_y):

                                    # Check if cursor is located inside the image border
                                    if (cursor_x_pos >= image_border_x_start) and (cursor_x_pos <= image_border_x_end):
                                        if (cursor_y_pos >= image_border_y_start) and (cursor_y_pos <= image_border_y_end):

                                            # Add items to a list that user hover her/his mouse
                                            mouse_was_here.append((visible_item[0], 0))

            # Before moving on the next row, add interacted items to a new list.
            if len(mouse_was_here) > 0:
                mouse_was_here = list(set(mouse_was_here))
                raw_df.loc[counter, 'good_catalog_items'] = [mouse_was_here]
            else:
                raw_df.loc[counter, 'good_catalog_items'] = 'No Item Found'

            counter += 1
        else:
            raw_df.loc[counter, 'good_catalog_items'] = 'No Item Found'
            counter += 1

    log_event('Method get_interacted_catalog_items() finished..')

    return raw_df

def map_items_to_dict(save_dict):
    """
        Method is responsible mapping current items in the database to a dictionary.

        1. Fetch all distinct items from the database.
        2. Create a dictionary.
        3. Iterate on the fetched items and append the item to dictionary starting from zero to len(distinct_items)

        :return
            saved pickle file represents Dictionary: { index:item_id1, index2:item_id2 .... }
    """
    # If item dictionary exists, load it to memory
    if not save_dict:
        item_dict_path = f'{path_dictionary["path_item_dictionary"]}'
        item_dict_file = open(item_dict_path, 'rb')
        return pickle.load(item_dict_file)

    # Query all visited items and put into a dataframe
    visited_items = DbHelper.fetch_all_items()
    visited_items_columns = ['item_id']
    visited_items_df = pd.DataFrame(data=visited_items, columns=visited_items_columns)
    item_list = visited_items_df['item_id'].values.tolist()

    # Query all the items shown in the catalog page and put into a dataframe
    catalog_items = DbHelper.get_all_catalog_items_less_columns()
    catalog_items_columns = ['user_id', 'item_id', 'session_id', 'catalog_item_list']
    catalog_items_df = pd.DataFrame(data=catalog_items, columns=catalog_items_columns)
    catalog_items_df['catalog_item_list'] = catalog_items_df.apply(clean_objects_listed, axis=1)

    catalog_items_list = catalog_items_df['catalog_item_list'].values.tolist()

    # iterate all the catalog items and add them to a list.
    for item_information in catalog_items_list:  # item information is a list contains ['itemID', 'x_coordinate', 'y_coordinate']
        if item_information != 'No Catalog Item':
            for item in item_information:
                item_list.append(item[0])

    # Remove the duplicates
    item_list = list(set(item_list))

    item_dict = {}
    counter = 0
    for item in item_list:
        item_dict[counter] = item
        counter += 1

    if save_dict:
        file_handler = open(f'{path_dictionary["path_item_dictionary"]}', "wb")
        pickle.dump(item_dict, file_handler)
        file_handler.close()

    return item_dict

def create_catalog_dataframe(save_dataframe):
    """
    Method is responsible for pre-processing the user sequences.

    Workflow:
        1. Query for all the user visits from the database which has the pageType = 'katalog' OR pageType = 'index' OR pageType = 'informace' and
           put them into a dataframe.
        2. Pre-process the necessary column data:
            2.1 Call clean_objects_listed method the pre-process catalog_item_list column.
            2.2 Call clean_logs method the pre-process user_log_list column.
            2.4 Call get_interacted_catalog_items to get the list of items in the catalog page that user has interaction.
            2.3 Call label_page_type method the pre-process catalog_item_list column.
        3. Drop unnecessary columns.
        4. Save the dataframe and csv to file system

    @param save_dataframe: Boolean to save the dataframe
    @return: Dataframe and .csv file is saved to the file system
    """
    if not save_dataframe:
        return pd.read_pickle(f'{path_dictionary["catalog_dataframe_grouped_path"]}')

    catalog_items = DbHelper.get_all_catalog_items()

    # Create dataframe to put all information together
    columns = ['user_id', 'item_id', 'session_id', 'window_size_x', 'window_size_y', 'page_size_x', 'page_size_y', 'catalog_item_list', 'user_log_list']
    catalog_items_df = pd.DataFrame(catalog_items, columns=columns)

    # Clean 'Catalog Items' that user see during a session
    catalog_items_df['catalog_item_list'] = catalog_items_df.apply(clean_objects_listed, axis=1)

    # Clean Log Files
    catalog_items_df['user_log_list'] = catalog_items_df.apply(clean_logs, axis=1)

    # Get Catalog Items that user hover or has a click action, her/his mouse
    catalog_items_df = get_interacted_catalog_items(catalog_items_df)

    # Label the catalog items as 0
    catalog_items_df['catalog_item_list'] = catalog_items_df.apply(label_page_type, axis=1)

    catalog_items_df_grouped = catalog_items_df.groupby(['user_id', 'session_id'], as_index=False).agg(lambda x: list(x))
    catalog_items_df_grouped.drop(['item_id', 'window_size_x', 'window_size_y', 'page_size_x', 'page_size_y'], axis=1, inplace=True)

    if save_dataframe:
        catalog_items_df.to_pickle(f'{path_dictionary["path_raw_catalog_dataframe"]}')
        catalog_items_df_grouped.to_pickle(f'{path_dictionary["path_catalog_dataframe"]}')
        catalog_items_df_grouped.to_csv(f'{path_dictionary["path_catalog_csv"]}', index=False, sep='|')
    return catalog_items_df_grouped

def create_user_sequence_dataframe(save_dataframe):
    """
    Method is responsible for pre-processing the user sequences.

    Workflow:
        1. Query for all the user visits from the database which will return one row for each item-visit and put them into a Dataframe.
        2. Group the user visits by using 'user_id' and 'session_id' columns so that we can user visits happen in the same session together.
        3. Once we have the individual item-visits aggragated to each other, rename the 'item_id' column to 'item_sequnce'
        4. Create a new column which will contain the sequence length. Then calculate the length of each user sequence and insert it to the 'sequence_length' column.
        2. Split the raw logs by newline. This will put each user action to a list
        3. Create a new list to insert cleaned actions. Then, iterate the user actions:
            |--> Replace the defined actions names 'MouseMove, to:', 'Scroll, to:', ' MouseClick' to 'MM', 'MS', and 'MC' respectively.
            |--> Then append actions to a new list in ['action_type', 'x_coordinate', 'y_coordinate'] structure.

    @param save_dataframe: Boolean to save the dataframe
    @return: Dataframe and .csv file is saved to the file system
    """
    if not save_dataframe:
        return pd.read_pickle(f'{path_dictionary["path_user_sequence_dataframe"]}')

    columns = ['user_id', 'item_id', 'session_id', 'session_start_time']

    items_visited_list = DbHelper.fetch_user_item_seq_info_with_date()
    user_item_visit_df = pd.DataFrame(data=items_visited_list, columns=columns)
    user_item_visit_df['session_start_time'] = pd.to_datetime(user_item_visit_df['session_start_time'], errors = 'coerce')

    item_sequence_df = user_item_visit_df.groupby(['user_id', 'session_id'], as_index=False).agg(lambda x: list(x))
    item_sequence_df.rename(columns={'item_id': 'item_sequence'}, inplace=True)

    # Remove the items_visited_list for memory management
    del items_visited_list, user_item_visit_df

    item_sequence_df['sequence_length'] = item_sequence_df['item_sequence'].str.len()

    item_sequence_df = item_sequence_df.drop(['sequence_length'], axis=1)
    item_sequence_df = item_sequence_df.reset_index(drop=True)

    if save_dataframe:
        item_sequence_df.to_pickle(f'{path_dictionary["path_user_sequence_dataframe"]}')
        item_sequence_df.to_csv(f'{path_dictionary["path_user_sequence_csv"]}', index=False, sep='|')

    return item_sequence_df

def merge_dataframes(df1, df2, save_dataframe):
    """
    Method is responsible for merging the given dataframes. Once the dataframes are merged, we must apply some pre-processing.

    Workflow:
        1. Merge df1 and df2 together. Apply the merge on 'left'
        2. Iterate on 'catalog_item_list', 'good_catalog_items' and 'user_log_list' then replace the null values with
           'No Catalog Found', 'No Item Found', and 'No log Found' respectively.
        3. Remove the catalog visits from end of the user sequences
        4. Delete the row labeled as 'To Delete' after calling the remove_last_catalog method.
        5. Create a new column named 'sequence_length' and insert the sequence lengths to the column.
        6. Remove every user sequence which has less than 2 visits. (Because we cannot make next-item recommendations unless we don't know the next
           item in the sequence...)
        7. Reset the index of the database since we drop many rows.
        8. Save the dataframe to file system.

    @param df1: user_sequence_dataframe
    @param df2: catalog_dataframe
    @param save_dataframe: Boolean to save the dataframe

    @return: Dataframe and .csv file is saved to the file system
    """
    merged_dataframe = pd.merge(df1, df2, on=['user_id', 'session_id'], how='left')

    for row in merged_dataframe.loc[merged_dataframe.catalog_item_list.isnull(), 'catalog_item_list'].index:
        merged_dataframe.at[row, 'catalog_item_list'] = 'No Catalog Item'

    for row in merged_dataframe.loc[merged_dataframe.good_catalog_items.isnull(), 'good_catalog_items'].index:
        merged_dataframe.at[row, 'good_catalog_items'] = 'No Item Found'

    for row in merged_dataframe.loc[merged_dataframe.user_log_list.isnull(), 'user_log_list'].index:
        merged_dataframe.at[row, 'user_log_list'] = 'No log found'

    # Check if the last item in the user seq is a catalog page
    merged_dataframe = merged_dataframe.apply(remove_last_catalog, axis=1)
    merged_dataframe = merged_dataframe[merged_dataframe.item_sequence != 'To Delete']
    merged_dataframe = merged_dataframe.reset_index(drop=True)

    merged_dataframe['sequence_length'] = merged_dataframe['item_sequence'].str.len()

    merged_dataframe = merged_dataframe.drop(merged_dataframe[merged_dataframe.sequence_length < 2].index)
    merged_dataframe = merged_dataframe.reset_index(drop=True)

    if save_dataframe:
        merged_dataframe.to_pickle(f'{path_dictionary["path_merged_dataframe"]}')
        merged_dataframe.to_csv(f'{path_dictionary["path_merged_csv"]}', index=False, sep='|')

    return merged_dataframe

"""
    When new dataset arrives:
        1. Create a new item dictionary so we can have all the items mapped.
        2. Create a raw dataframe that represents user logs, catalog items and good catalog items.
            user_logs = Shows the user actions
            catalog_items = Shows the items that were represented to user in the catalog page
            good_catalog_items = Shows the catalog items that user have interaction (click and mouse hover)
        3. Create a raw dataframe that represents user sequences.
        4. Create a new dataframe and merged the user sequences and catalog items on it.
        
    Once the merged dataframe is created we will have a dataframe which contains:
        -> User sequence (list of visited pages in a session)
        -> Catalog items (list of items that were shown in the catalog)
        -> Good Catalog Items (list of items that were shown in the catalog and user had interaction)
        -> User Logs (List of implicit feedback)        
"""

path_dictionary = {'path_user_sequence_dataframe': '../source_data/user_sequence_dataframe.pickle',
                   'path_user_sequence_csv': '../source_data/user_sequence_dataframe.csv',
                   'path_item_dictionary': '../source_data/item.dictionary',
                   'path_catalog_dataframe': '../source_data/catalog_dataframe.pickle',
                   'path_catalog_csv': '../source_data/catalog_dataframe.csv',
                   'path_raw_catalog_dataframe': '../source_data/catalog_dataframe_raw.pickle',
                   'path_merged_dataframe': '../source_data/merged_dataframe.pickle',
                   'path_merged_csv': '../source_data/merged_dataframe.csv'}

# In order to create and save the files make the boolean parameters True. When False, code will try to load the data directly from the respective folder.

log_event('| Creating the Item Dictionary..')
item_dictionary = map_items_to_dict(save_dict=False) # --> Item Dictionary is located in the source folder since given partial database do not contains
                                                     # all items from the source database.
log_event('| --> Dictionary is created..')

log_event('| Creating the Catalog Dataframe..')
catalog_df = create_catalog_dataframe(save_dataframe=True)
log_event('| --> Catalog Dataframe is created..')

log_event('| Creating the User Sequence Dataframe..')
user_sequence_df = create_user_sequence_dataframe(save_dataframe=True)
log_event('| --> User Sequence Dataframe is created..')

log_event('| Merging the Dataframes..')
merged_df = merge_dataframes(df1=user_sequence_df, df2=catalog_df ,save_dataframe=True)
log_event('| --> Dataframes are merged..')

