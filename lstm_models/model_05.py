from model_evaluation.SequenceEvaluation import Evaluator
from lstm_models.parent_lstm_model import *

class LstmV5(BaseLSTM):
    def __init__(self, model_alias, train_alias, path_dictionary, training_dictionary, hyper_parameters):

        super().__init__(model_alias=model_alias,
                         train_alias=train_alias,
                         path_dictionary=path_dictionary,
                         training_dictionary=training_dictionary,
                         hyper_parameters=hyper_parameters)

        self.catalog_items = None
        self.good_catalog_items = None

        self.constant_for_catalog_item = 2.0
        self.constant_for_visible_item = 5.0
        self.constant_for_click_event = 15.0
        self.l2_norm_constant = 0.

    def train_model(self):
        """
            Main method to handle necessary information and folder creation, pre-processing and training the model.

            1. Check if the dedicated model folder exists. If not, create the folder.
            2. Create a pickle file which contains metadata information of the model.
            3. If there exists an RNN Model created earlier then load the model and return it for the evaluation.
            4. If there is no trained RNN Model:
                4.1 Create a loop to iterate through all the split data. For each split data:
                    4.1.1 Create the respective directory.
                    4.1.2 Call process_training_data method for processing the train and test data and save the processed data to model folder.
                    4.1.3 Start Hyper Parameter Tuning.
            5. Find the best model from all the split trainings and use it for the evaluation.

        :return: Best RNN Model
        """

        if not self.is_exist(self.path_model_directory):
            # Then create the parent folder
            os.makedirs(self.path_model_directory)

            # Then create a meta-data pickle for the model
            self.create_meta_data_pickle()

        # Necessary meta-data file must be created before starting the training. Check if the file exists
        if self.is_exist(self.path_model_metadata):

            # We do not need to train a model if there is already a best model for the same training exist
            try:
                self.model = load_model(self.path_best_model)
                return
            except:
                self.log_event('There is no best trained model found in the parent folder. Going with the training...')

            # Load the model meta-data
            self.load_model_metadata()

            # Iterate trough the split data for the training
            for split_number in range(self.k_split):
                split_path = f'split_{str(split_number)}/'
                split_directory = self.path_model_directory + split_path

                # Check the split directory is already created. If it is, then we can directly start the training by using the existing data
                if self.is_exist(split_directory):
                    try:
                        self.load_best_tuned_model(split_number)
                    except (IndexError, FileNotFoundError):
                        self.load_fold_k_data_and_fit(split_number=int(split_number))

                else:
                    # Create a folder for the split data and prepare the data for the training
                    os.makedirs(split_directory)

                    # Create an array which will contain train features-labels and test features-labels
                    train_array = np.full(4, fill_value=self.mask_value, dtype=object)
                    train_index = 0
                    for position, split_name in enumerate(['train_split_', 'test_split_']):
                        training_features_directory = split_directory + f'{split_name}{str(split_number)}_all_training_features.data'
                        training_targets_directory  = split_directory + f'{split_name}{str(split_number)}_all_training_targets.data'
                        fold_directory = self.path_shared_folds + f'{split_name}{str(split_number)}.fold'

                        self.process_training_data(fold_directory=fold_directory)

                        self.save_data_to_disk(data_to_save=self.all_features, path_to_save=training_features_directory)
                        train_array[train_index] = self.all_features
                        train_index += 1
                        self.all_features = None # Memory Management

                        self.save_data_to_disk(data_to_save=self.all_targets, path_to_save=training_targets_directory)
                        train_array[train_index] = self.all_targets
                        train_index += 1
                        self.all_targets = None # Memory Management

                    # Assign the input data to respective variables for the training
                    self.train_features = train_array[0]
                    self.train_targets = train_array[1]
                    self.test_features = train_array[2]
                    self.test_targets = train_array[3]
                    del train_array

                    self.start_hyper_parameter_tuning(split_number)

            self.retrieve_best_model(metric=self.hyper_parameters['metric'])

    def process_training_data(self, fold_directory):
        """
            Method is responsible of creating an input data which is ready to be fit to RNN.
            1. Load the split data and transform required columns to a list.
            2. Create Features and Targets
            3. Encode the Features
            4. Apply Mask on the input features
            5. One-Hot-Encode the Targets.

            (Navigate to respective method for its own comments)

        :param fold_directory: The directory of the fold data
        """
        # Step 1: Load the fold data as the user sequences
        self.user_sequences_dataframe = joblib.load(fold_directory)

        # Get the necessary lists from the dataframe for processing
        self.user_sequence_list = self.user_sequences_dataframe['item_sequence'].values
        self.catalog_items      = self.user_sequences_dataframe['catalog_item_list'].values
        self.good_catalog_items = self.user_sequences_dataframe['good_catalog_items'].values

        # Step 2: Create Features and Targets
        self.create_features_and_targets()

        # Step 3: Encode Features
        self.encode_features()

        # Step 4: Apply Mask to all Features
        self.apply_mask(all_features=self.all_features, number_of_features=self.number_of_distinct_items)

        # Step 5: Encode Targets
        self.encode_targets()

    def encode_features(self):
        """
            Method will encode the features based on their page type. If the user visited a dedicated tour, then tourID will be One-Hot-Encoded.
            If user visited a category page, then method will find all the tours shown in the catalog page, categorize and scale them by their
            visibility to the user and if the user clicked on a an item. Result vector will be K-Hot-Encoded.

            1. Iterate through self.all_features.
                -> If clicked object is a dedicated tour, then One-Hot-Encode the objectID
                -> Else, user visited a catalog page. K-Hot-Encode the items shown to user. (Navigate to encode_catalog method for more details)
        """
        user_count = 0
        for individual_sequence in self.all_features:
            encoded_sequence = np.full(shape=(len(individual_sequence), self.number_of_distinct_items), fill_value=self.mask_value, dtype='float32')
            catalog_counter = 0

            for idx, item in enumerate(individual_sequence):
                if item != 0:
                    encoded_sequence[idx] = self.encode_item(item_id=item)
                else:
                    encoded_sequence[idx], catalog_counter = self.encode_catalog(catalog_items_for_the_session=self.catalog_items[user_count],
                                                                                 good_catalog_item_for_the_session=self.good_catalog_items[user_count],
                                                                                 catalog_counter=catalog_counter)

            self.all_features[user_count] = encoded_sequence

            user_count += 1

        self.log_event('   -> K Hot Encoding the features is over..')

        del self.good_catalog_items, self.catalog_items
        self.log_event('   ----> self.good_catalog_items, self.catalog_items objects are deleted for memory management..!')
        self.log_event('   -> K Hot Encoding is over..')

    def encode_catalog(self, catalog_items_for_the_session, good_catalog_item_for_the_session, catalog_counter):
        """
            Method is responsible to K-Hot-Encode catalog items
            1. Iterate through the items that were shown to user during the catalog visit.
            2. First, create one catalog vector and One-Hot-Encode the item.
            3. Then, check the other catalog items if user had some interaction.
                3.1 For all the catalog items user interacted, insert respective action constant to the created catalog vector.
                3.2 Scale down the catalog vector by using l2-normalization

        :param catalog_items_for_the_session: All the shown items in the catalog page during the respective user session
        :param good_catalog_item_for_the_session: All the catalog items and their user interaction type.
        :param catalog_counter: The number of the catalog counter so far.
        :return: K-Hot-Encoded Catalog vector, catalog counter
        """
        catalog_vector = np.zeros(self.number_of_distinct_items, dtype='float32')

        if isinstance(catalog_items_for_the_session, list):
            corresponding_catalog_page = catalog_items_for_the_session[catalog_counter]
            if corresponding_catalog_page != 'No Catalog Item' and corresponding_catalog_page != 'No Item Found':
                for cat_item in corresponding_catalog_page:
                    cat_item_id = cat_item[0]
                    encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(cat_item_id)]
                    catalog_vector[int(encoding)] = encoding

                if isinstance(good_catalog_item_for_the_session, list):
                    corresponding_good_catalog_items = good_catalog_item_for_the_session[catalog_counter]
                    if corresponding_good_catalog_items != 'No Item Found':
                        for good_item in corresponding_good_catalog_items:
                            good_item_id = good_item[0]
                            status = good_item[1]
                            encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(good_item_id)]
                            if status == 1:
                                catalog_vector[encoding] = self.constant_for_click_event
                            if status == 0 and catalog_vector[encoding] != self.constant_for_click_event:
                                catalog_vector[encoding] = self.constant_for_visible_item

                temp = [0.0, 1.0, self.constant_for_catalog_item, self.constant_for_visible_item, self.constant_for_click_event]
                catalog_vector = np.array([self.constant_for_catalog_item if item not in temp else item for item in catalog_vector])

                # Scale down the values to have sum of vector == 1
                catalog_vector = self.scale_down_vector(vector_to_calc_l2=catalog_vector)

                catalog_counter += 1
                return catalog_vector, catalog_counter

        catalog_counter += 1
        return np.full(shape=self.number_of_distinct_items, fill_value=self.mask_value, dtype='float32'), catalog_counter

    def prepare_evaluation(self):
        """
             Method is responsible of creating an evaluation data suitable for the model.
             1. Check if the evaluation data exists. If yes, then call the evaluation class.
             2. If training data does not exist, create one:
                 2.1 Load the evaluation dataframe from the source folder
                 2.2 Create a new column to hold raw sequence information so that we can see which vector refers which item.
                 2.3 Create the suitable feature encodings
                 2.4 Drop the unnecessary columns from the database and keep only user_id, session_id, item_sequence and decoded_item_sequence columns.
                 2.5 Transform the columns to a list and then save it to the model folder for future trainings
                 2.6 Load the model metadata and call the evaluation class.
         """
        self.model = load_model(self.path_best_model)

        if not self.is_exist(f'{self.path_model_directory}evaluation_ready_sequences.data'):
            self.evaluation_dataframe = self.load_evaluation_data()

            self.user_sequence_list = self.evaluation_dataframe['item_sequence'].values
            self.catalog_items = self.evaluation_dataframe['catalog_item_list'].values
            self.good_catalog_items = self.evaluation_dataframe['good_catalog_items'].values

            self.evaluation_dataframe.drop(['catalog_item_list', 'session_start_time', 'good_catalog_items', 'sequence_length'], axis=1, inplace=True)

            self.evaluation_dataframe['decoded_item_sequence'] = self.evaluation_dataframe['item_sequence'].copy(deep=True)

            self.all_features = self.evaluation_dataframe['item_sequence'].values
            self.encode_features()
            self.evaluation_dataframe['item_sequence'] = self.all_features

            sequences = self.evaluation_dataframe.values.tolist()
            joblib.dump(sequences, f'{self.path_model_directory}evaluation_ready_sequences.data')
        else:
            sequences = joblib.load(f'{self.path_model_directory}evaluation_ready_sequences.data')
            self.load_model_metadata()

        self.evaluation_creator(sequences_to_evaluate=sequences, evaluation_class=Evaluator)