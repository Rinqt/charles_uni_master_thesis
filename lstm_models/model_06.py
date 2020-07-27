from model_evaluation.SequenceEvaluationTwoBranch import Evaluator
from lstm_models.parent_lstm_model import *
import gc

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model

class LstmV6(BaseLSTM):
    def __init__(self, model_alias, train_alias, path_dictionary, training_dictionary, hyper_parameters):

        super().__init__(model_alias=model_alias,
                         train_alias=train_alias,
                         path_dictionary=path_dictionary,
                         training_dictionary=training_dictionary,
                         hyper_parameters=hyper_parameters)

        self.auxiliary_dataframe = None
        self.auxiliary_all_features = None
        self.auxiliary_all_targets = None
        self.auxiliary_data_feature_list = None
        self.auxiliary_data_train_feature_list = None
        self.auxiliary_data_test_feature_list = None
        self.auxiliary_data_train_target_list = None
        self.auxiliary_data_test_target_list = None
        self.auxiliary_vector_size = 0

        self.user_sequence_start = None
        self.catalog_items = None
        self.good_catalog_items = None

        self.constant_for_catalog_item = 2.0
        self.constant_for_visible_item = 5.0
        self.constant_for_click_event = 15.0
        self.l2_norm_constant = 0.

    def train_model(self):
        """
            Main method to handle necessary information and folder creation, pre-processing and training the model.

            1. Load the necessary auxiliary data and set self.auxiliary_vector_size to object metadata vector size
            2. Check if the dedicated model folder exists. If not, create the folder.
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

        # Load Auxiliary Data
        self.auxiliary_dataframe = pd.read_pickle(f'D:/Thesis/recsys_thesis/data/auxiliary_df.pickle')

        # Set the auxiliary vector size
        self.auxiliary_vector_size = len(self.auxiliary_dataframe.iloc[0].all_vectors)

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
            self.encoding_vector_size = self.number_of_distinct_items

            # Iterate trough the split data for the training
            for split_number in range(self.k_split):
                split_path = f'split_{str(split_number)}/'
                split_directory = self.path_model_directory + split_path

                # Check the split directory is already created. If it is, then we can directly start the training by using the existing data
                if self.is_exist(split_directory):
                    try:
                        self.load_best_tuned_model(split_number)
                    except (IndexError, FileNotFoundError):
                        self.auxiliary_data_train_feature_list = joblib.load(f'{self.path_model_directory}split_{str(split_number)}/train_split_{str(split_number)}_auxiliary_all_features.data')
                        self.auxiliary_data_test_feature_list = joblib.load(f'{self.path_model_directory}split_{str(split_number)}/test_split_{str(split_number)}_auxiliary_all_features.data')
                        self.load_fold_k_data_and_fit(split_number=int(split_number))
                        gc.collect()
                else:
                    # Create a folder for the split data and prepare the data for the training
                    os.makedirs(split_directory)

                    # Create an array which will contain train features-labels and test features-labels
                    train_array = np.full(4, fill_value=self.mask_value, dtype=object)
                    auxiliary_data_array = np.full(2, fill_value=self.mask_value, dtype=object)

                    train_index = 0
                    au_index = 0
                    for position, split_name in enumerate(['train_split_', 'test_split_']):
                        training_features_directory = split_directory + f'{split_name}{str(split_number)}_all_training_features.data'
                        training_targets_directory  = split_directory + f'{split_name}{str(split_number)}_all_training_targets.data'
                        training_auxiliary_data_directory  = split_directory + f'{split_name}{str(split_number)}_auxiliary_all_features.data'
                        fold_directory = self.path_shared_folds + f'{split_name}{str(split_number)}.fold'

                        self.process_training_data(fold_directory=fold_directory)

                        self.save_data_to_disk(data_to_save=self.all_features, path_to_save=training_features_directory)
                        train_array[train_index] = self.all_features
                        train_index += 1
                        self.all_features = None  # Memory Management

                        self.save_data_to_disk(data_to_save=self.all_targets, path_to_save=training_targets_directory)
                        train_array[train_index] = self.all_targets
                        train_index += 1
                        self.all_targets = None  # Memory Management

                        self.save_data_to_disk(data_to_save=self.auxiliary_all_features, path_to_save=training_auxiliary_data_directory)
                        auxiliary_data_array[au_index] = self.auxiliary_all_features
                        au_index += 1
                        self.auxiliary_all_features = None  # Memory Management

                    # Call garbage collector to clean memory
                    gc.collect()

                    # Assign the input data to respective variables for the training
                    self.auxiliary_data_train_feature_list = auxiliary_data_array[0]
                    self.auxiliary_data_test_feature_list = auxiliary_data_array[1]
                    self.train_features = train_array[0]
                    self.train_targets = train_array[1]
                    self.test_features = train_array[2]
                    self.test_targets = train_array[3]
                    del train_array, auxiliary_data_array

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
        self.catalog_items = self.user_sequences_dataframe['catalog_item_list'].values
        self.good_catalog_items = self.user_sequences_dataframe['good_catalog_items'].values
        self.user_sequence_start = self.user_sequences_dataframe['session_start_time'].values

        # Step 2: Create Features and Targets
        self.create_features_and_targets()

        # Step 3: Encode Features
        self.encode_features()

        # Step 4: Apply Mask to all Features
        self.all_features = self.apply_mask(all_features=self.all_features, number_of_features=self.number_of_distinct_items)

        # Step 5: Apply Mask to all Auxiliary Data Features
        self.auxiliary_all_features = self.apply_mask(all_features=self.auxiliary_all_features, number_of_features=self.auxiliary_vector_size)

        # Step 6: Encode Targets
        self.encode_targets()

    def encode_features(self):
        """
            Method will encode the features based on their page type. If the user visited a dedicated tour, then tourID will be One-Hot-Encoded.
            If user visited a category page, then method will find all the tours shown in the catalog page, categorize and scale them by their
            visibility to the user and if the user clicked on a an item. Result vector will be K-Hot-Encoded.

            Meanwhile it will also find the related object metadata to be used as the second input for the RNN

            1. Iterate through self.all_features.
                -> If clicked object is a dedicated tour, then One-Hot-Encode the objectID and find the respective content metadata
                -> Else, user visited a catalog page. K-Hot-Encode the items shown to user and find the respective content metadata.
                (Navigate to encode_catalog method for more details)
        """
        self.log_event('   -> K Hot Encoding the features..')

        self.auxiliary_all_features = np.zeros(len(self.all_features), dtype=object)

        user_count = 0
        for individual_sequence in self.all_features:
            encoded_sequence = np.full(shape=(len(individual_sequence), self.number_of_distinct_items), fill_value=self.mask_value, dtype='float32')
            encoded_auxiliary_info = np.full(shape=(len(individual_sequence), self.auxiliary_vector_size), fill_value=self.mask_value, dtype='float32')

            catalog_counter = 0
            for idx, item in enumerate(individual_sequence):
                if item != 0:
                    encoded_sequence[idx] = self.encode_item(item_id=item)

                    # Find corresponding item information
                    encoded_auxiliary_info[idx] = self.find_auxiliary_data(item=item, seq_start=self.user_sequence_start[user_count][idx])

                else:
                    encoded_sequence[idx], encoded_auxiliary_info[idx], catalog_counter = self.encode_catalog(catalog_items_for_the_session=self.catalog_items[user_count],
                                                                                                              good_catalog_item_for_the_session=self.good_catalog_items[user_count],
                                                                                                              catalog_counter=catalog_counter,
                                                                                                              user_count=user_count,
                                                                                                              idx=idx)

            self.all_features[user_count] = encoded_sequence
            self.auxiliary_all_features[user_count] = encoded_auxiliary_info

            user_count += 1

        self.log_event('   -> K Hot Encoding the features is over..')

    def find_auxiliary_data(self, item, seq_start):
        """
            Method will find which tour that user interacted and scale the information to be used as auxiliary data.
            1. Find the possible tours that user might interacted by using their item_id.
            2. Filter the found tours to get only the tours that satisfies the timeline of the user sequence start
            3. Used the valid_from and valid_to columns to filter the tours even more to identify specific tour.
            4. Average the tour information and return the vector.
        :param item: The tour item
        :param seq_start: Timestamp of the user sequence start
        :return: One vector which contains information regarding the all possible tours related with one tour series that user interacted with.
        """
        # Find the auxiliary input for the respective input
        respective_df_rows = self.auxiliary_dataframe[self.auxiliary_dataframe['item_id'] == item]

        if len(respective_df_rows) == 0:
            return np.zeros(self.auxiliary_vector_size, dtype='float32')

        correct_time_stamp_item = respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date())]

        if len(correct_time_stamp_item) > 0:
            if len(correct_time_stamp_item[correct_time_stamp_item['valid_to'].isnull()]) > 0:
                return self.get_mean_of_vectors(correct_time_stamp_item[correct_time_stamp_item['valid_to'].isnull()])
            else:
                if len(respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date()) & (seq_start.date() <= respective_df_rows['valid_to'])]) > 0:
                    # Take the mean of the all item vectors
                    all_possible_items = respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date()) & (seq_start.date() <= respective_df_rows['valid_to'])]
                    return self.get_mean_of_vectors(all_possible_items)

        return self.get_mean_of_vectors(respective_df_rows)

    def get_mean_of_vectors(self, items):
        # Create item vectors
        item_vectors = items['all_vectors'].to_numpy()

        # Calculate the mean of the vectors and return it
        return item_vectors.sum() / len(item_vectors)

    def encode_catalog(self, catalog_items_for_the_session, good_catalog_item_for_the_session, catalog_counter, user_count, idx):
        """
            Method is responsible to K-Hot-Encode catalog items in the user sequence and content metadata in the auxiliary data
            1. Create one vector to hold catalog data and one vector to hold content metadata.
            2. Iterate through the items that were shown to user during the catalog visit.
            3. First, create one catalog vector and One-Hot-Encode the item.
                3.1 Find the tour descriptions of the encoded tour by using find_auxiliary_data method.
            4. Then, check the other catalog items if user had some interaction.
                4.1 For all the catalog items user interacted, insert respective action constant to the created catalog vector.
                4.2 Find the tour descriptions of the encoded tours by using find_auxiliary_data method.
                4.3 Scale down the catalog vector by using l2-normalization.
                4.4 Scale down the auxiliary vector by using l2-normalization.

        :param catalog_items_for_the_session: All the shown items in the catalog page during the respective user session
        :param good_catalog_item_for_the_session: All the catalog items and their user interaction type.
        :param catalog_counter:
        :param user_count: The number of the catalog counter so far.
        :param idx: The index of the item in the user sequence
        :return: K-Hot-Encoded Catalog vector, K-Hot-Encoded Auxiliary vector, catalog counter
        """
        catalog_vector = np.zeros(self.number_of_distinct_items, dtype='float32')
        catalog_auxiliary_info = np.zeros(self.auxiliary_vector_size, dtype='float32')

        if isinstance(catalog_items_for_the_session, list):
            corresponding_catalog_page = catalog_items_for_the_session[catalog_counter]
            if corresponding_catalog_page != 'No Catalog Item' and corresponding_catalog_page != 'No Item Found':
                for cat_item in corresponding_catalog_page:
                    cat_item_id = cat_item[0]
                    encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(cat_item_id)]
                    catalog_vector[int(encoding)] = encoding

                    # Find corresponding item information
                    catalog_auxiliary_info += self.find_auxiliary_data(item=cat_item_id, seq_start=self.user_sequence_start[user_count][idx])

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

                            # Find corresponding item information
                            catalog_auxiliary_info += self.find_auxiliary_data(item=good_item_id, seq_start=self.user_sequence_start[user_count][idx])

                temp = [0.0, 1.0, self.constant_for_catalog_item, self.constant_for_visible_item, self.constant_for_click_event]
                catalog_vector = np.array([self.constant_for_catalog_item if item not in temp else item for item in catalog_vector])

                # Scale down the values to have sum of vector == 1
                catalog_vector = self.scale_down_vector(vector_to_calc_l2=catalog_vector)
                catalog_auxiliary_info = self.scale_down_vector(vector_to_calc_l2=catalog_auxiliary_info)

                catalog_counter += 1
                return catalog_vector, catalog_auxiliary_info, catalog_counter

        catalog_counter += 1
        return np.full(shape=self.number_of_distinct_items, fill_value=self.mask_value, dtype='float32'), np.full(shape=self.auxiliary_vector_size, fill_value=self.mask_value, dtype='float32'), catalog_counter

    def apply_mask(self, all_features, number_of_features):
        """
            Method creates a 3D numpy array to be used to fit RNN. All the data was staying in one-dimension vector so far. However, it needs to be converted to 3D array in order
            to be fit into RNN. Structure of the array [# of examples, timestamps, features]

            1. Create an empty 3D numpy array using the number of sequences, maximum sequence size and features vector size
            2. Iterate through all the features
                2.1 Check if the sequence length is not exceeding the maximum_sequence_size.
                    2.1.1 If does, calculate the difference and get the items in the sequence starting from the difference (e.g. sequence = sequence[difference:]
                2.2 Fill the 2nd and 3rd dimension of the array with the sequence information

        :param all_features: Features to transform to 3D Array
        :param number_of_features: Encoding vector size
        """
        self.log_event('|----> Masking the Feature List..')

        masked_features = np.full((len(all_features), self.max_sequence_size, number_of_features), fill_value=self.mask_value, dtype='float32')

        for s, x in enumerate(all_features):
            seq_len = x.shape[0]
            difference = seq_len - self.max_sequence_size

            if difference > 0:
                x = x[difference:]
                seq_len = seq_len - difference

            if masked_features.shape[2] == 1:
                masked_features[s, -seq_len:, :] = x[:seq_len].reshape(seq_len, 1)
            else:
                masked_features[s, -seq_len:, :] = x[:seq_len]

        self.log_event('|----> Masking is completed..')
        return masked_features

    def start_hyper_parameter_tuning(self, split_number):
        """
            Method is responsible to find best RNN model fit the split data. This is an override method since Model 2 has a
            different architecture than all the other models

            1. Use Keras Tuner to find best possible configuration for the RNN Model
            2. Train the candidate models
            3. Find the best performed model and train the model on the split data.
            4. After the training;
                4.1 Plot and save the model architecture to the file system.
                4.2 Save the RNN Model to the file system
                4.3 Save the training data (plots and csv) to the file system
            5. Append the best performed model to best_possible_models list to compare all the models easily after all the split training is over.

        :param split_number: The number of the split data
        """
        self.log_event('Training with HyperParameter Tuning is started..')

        tuner = RandomSearch(self.build_model,
                             objective=self.hyper_parameters['objective_function'],
                             max_trials=self.max_trials,
                             executions_per_trial=self.executions_per_trial,
                             seed=self.random_state,
                             project_name=f'split_{str(split_number)}',
                             directory=os.path.normpath(self.path_tuner_directory))

        tuner.search(x=[self.train_features, self.auxiliary_data_train_feature_list],
                     y=self.train_targets,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     verbose=2,
                     validation_data=([self.test_features, self.auxiliary_data_test_feature_list], self.test_targets))

        # Get the trials
        trials = tuner.oracle.trials
        best_model = tuner.oracle.get_best_trials()[0].trial_id

        self.model = tuner.get_best_models(num_models=1)[0]

        self.model_history = self.model.fit(x=[self.train_features, self.auxiliary_data_train_feature_list],
                                            y=self.train_targets,
                                            epochs=self.epochs,
                                            batch_size=self.batch_size,
                                            verbose=2,
                                            validation_data=([self.test_features, self.auxiliary_data_test_feature_list], self.test_targets))

        self.print_hyper_parameter_results(split_number=split_number,
                                           trials=trials,
                                           best_model=best_model)

        keras.utils.plot_model(self.model, to_file=f'{self.path_model_directory}{self.model_alias}.png', show_shapes=True, show_layer_names=True)


        current_time = datetime.datetime.now()
        save_path = f'{self.path_tuner_directory}/split_{str(split_number)}/split_{str(split_number)}_{self.model_alias}_{str(current_time.strftime("%Y-%m-%d_%H-%M-%S"))}.h5'
        self.model.save(save_path)

        hist_df = pd.DataFrame(self.model_history.history)
        hist_df.to_csv(f'{self.path_tuner_directory}/split_{str(split_number)}/split_{str(split_number)}_{self.model_alias}_best_model_history.csv', index=False, header=True)
        self.best_possible_models.append(hist_df)

        self.plot_train_summary(plot_title=f'Split {str(split_number)}', split_number=str(split_number))

    def build_model(self, hp):
        """
            Model is responsible of building a RNN model with the decided parameters.
            Method will create two branch for the RNN. While one branch will have the user sequences as input,
            then other branch will have the respective auxiliary input data.
            After two branches is created, model will concatenate the layer, add one dense layer and compile the model

        :param hp: Keras Tuner Parameters
        :return: Functional Recurrent Neural Network Model
        """
        self.log_event('Building Version 6 LSTM Network..')

        sequences_model = self.create_sequence_branch(hp)
        auxiliary_model = self.create_sequential(hp)

        concat_layer = keras.layers.Concatenate()([sequences_model.output, auxiliary_model.output])

        tour_pred = keras.layers.Dense(self.number_of_distinct_items, activation=self.hyper_parameters['dense_activation'], name='sequence_prediction')(concat_layer)

        model = keras.Model(inputs=[sequences_model.input, auxiliary_model.input], outputs=tour_pred)

        opt = Adam(hp.Choice('learning_rate', values=self.hyper_parameters['learning_rate']))

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=self.hyper_parameters['metric'])

        self.log_event('    -> Returning the model.')

        return model

    def create_sequential(self, hp):
        """
            Model is responsible of building an RNN model with the decided parameters
        :param hp: Keras Tuner Parameters
        :return: Sequential Recurrent Neural Network Model
        """
        model = Sequential()

        model.add(Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.auxiliary_vector_size)))

        # Add One LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('first_layer',
                                    min_value=self.hyper_parameters['lstm_units']['min'],
                                    max_value=self.hyper_parameters['lstm_units']['max'],
                                    step=self.hyper_parameters['lstm_units']['step']),
                       return_sequences=True,
                       dropout=self.hyper_parameters['lstm_layer_dropout'],
                       recurrent_dropout=0.1,
                       activation=self.hyper_parameters['lstm_layer_activation']))

        model.add(BatchNormalization())

        # Add Dropout
        model.add(Dropout(hp.Choice('dropout_one', values=self.hyper_parameters['dropout'])))

        # Add the second LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('second_layer',
                                    min_value=self.hyper_parameters['lstm_units']['min'],
                                    max_value=self.hyper_parameters['lstm_units']['max'],
                                    step=self.hyper_parameters['lstm_units']['step']),
                       return_sequences=False,
                       dropout=self.hyper_parameters['lstm_layer_dropout'],
                       recurrent_dropout=0.1,
                       activation=self.hyper_parameters['lstm_layer_activation']))

        model.add(BatchNormalization())

        # Add Dropout
        model.add(Dropout(hp.Choice('dropout_one', values=self.hyper_parameters['dropout'])))

        # Add Output Layer
        #model.add(Dense(self.number_of_distinct_items, activation=self.hyper_parameters['dense_activation']))
        return model

    def create_sequence_branch(self, hp):
        """
            Model is responsible of building a FUNCTIONAL RNN model with the decided parameters
        :param hp: Keras Tuner Parameters
        :return: Functional Recurrent Neural Network Model
        """
        sequences_input = keras.layers.Input(shape=(self.max_sequence_size, self.number_of_distinct_items), name='sequences')

        sequences_features = Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.number_of_distinct_items))(sequences_input)
        sequences_features = LSTM(units=hp.Int('first_layer',
                                               min_value=self.hyper_parameters['lstm_units']['min'],
                                               max_value=self.hyper_parameters['lstm_units']['max'],
                                               step=self.hyper_parameters['lstm_units']['step']),
                                  return_sequences=True,
                                  dropout=self.hyper_parameters['lstm_layer_dropout'],
                                  recurrent_dropout=0.1,
                                  activation=self.hyper_parameters['lstm_layer_activation'])(sequences_features)

        sequences_features = BatchNormalization()(sequences_features)
        sequences_features = Dropout(hp.Choice('input_one_dropout_one', values=self.hyper_parameters['dropout']))(sequences_features)

        sequences_features = LSTM(units=hp.Int('second_layer',
                                               min_value=self.hyper_parameters['lstm_units']['min'],
                                               max_value=self.hyper_parameters['lstm_units']['max'],
                                               step=self.hyper_parameters['lstm_units']['step']),
                                  return_sequences=False,
                                  dropout=self.hyper_parameters['lstm_layer_dropout'],
                                  recurrent_dropout=0.1,
                                  activation=self.hyper_parameters['lstm_layer_activation'])(sequences_features)

        sequences_features = BatchNormalization()(sequences_features)
        sequences_features = Dropout(hp.Choice('input_one_dropout_one', values=self.hyper_parameters['dropout']))(sequences_features)

        model = Model(sequences_input, sequences_features)
        return model

    def prepare_evaluation(self):
        """
             Method is responsible of creating an evaluation data suitable for the model.
             1. Check if the evaluation data exists. If yes, then call the evaluation class.
             2. If training data does not exist, create one:
                 2.1 Load the evaluation dataframe and the auxiliary dataframe from the source folder.
                 2.2 Create a new column to hold raw sequence information so that we can see which vector refers which item.
                 2.3 Create the suitable feature encodings for the user sequences
                 2.4 Create the suitable feature encodings for the content metadata
                 2.5 Drop the unnecessary columns from the database and keep only user_id, session_id, item_sequence and decoded_item_sequence columns.
                 2.6 Transform the columns to a list and then save it to the model folder for future trainings
                 2.7 Load the model metadata and call the evaluation class.

                 Note that Model 6 has two inputs for the RNN Model. auxiliary_vector_size and encoding_vector_size values must reflect
                 the correct vector dimensions
         """
        self.model = load_model(self.path_best_model)

        if not self.is_exist(f'{self.path_model_directory}evaluation_ready_sequences.data'):
            self.evaluation_dataframe = self.load_evaluation_data()

            self.user_sequence_list = self.evaluation_dataframe['item_sequence'].values
            self.catalog_items = self.evaluation_dataframe['catalog_item_list'].values
            self.good_catalog_items = self.evaluation_dataframe['good_catalog_items'].values
            self.user_sequence_start = self.evaluation_dataframe['session_start_time'].values

            self.evaluation_dataframe.drop(['catalog_item_list', 'session_start_time', 'good_catalog_items', 'sequence_length'], axis=1, inplace=True)

            self.evaluation_dataframe['decoded_item_sequence'] = self.evaluation_dataframe['item_sequence'].copy(deep=True)

            self.all_features = self.evaluation_dataframe['item_sequence'].values

            self.auxiliary_dataframe = pd.read_pickle(f'D:/Thesis/recsys_thesis/data/auxiliary_df.pickle')
            self.auxiliary_vector_size = len(self.auxiliary_dataframe.iloc[0].all_vectors)

            self.encode_features()
            self.evaluation_dataframe['item_sequence'] = self.all_features
            self.evaluation_dataframe['auxiliary_features'] = self.auxiliary_all_features

            sequences = self.evaluation_dataframe.values.tolist()
            joblib.dump(sequences, f'{self.path_model_directory}evaluation_ready_sequences.data')
        else:
            sequences = joblib.load(f'{self.path_model_directory}evaluation_ready_sequences.data')
            self.load_model_metadata()
            self.auxiliary_dataframe = pd.read_pickle(f'D:/Thesis/recsys_thesis/data/auxiliary_df.pickle')
            self.auxiliary_vector_size = len(self.auxiliary_dataframe.iloc[0].all_vectors)
            del self.auxiliary_dataframe

        self.encoding_vector_size = self.number_of_distinct_items
        self.evaluation_creator(sequences_to_evaluate=sequences, evaluation_class=Evaluator)

    def multi_thread_sequence_evaluation(self, sequences_to_evaluate, evaluation_class):
        """
            Override the parent method for Model 6 since it is using 2 input for the RNN and it needs auxiliary_item_sequence
            to be sent to respective evaluation class.

        :param sequences_to_evaluate: List of the sequences to evaluate
        :param evaluation_class: Evaluation class
        """
        for user_id, session_id, user_sequence, decoded_individual_sequence, auxiliary_item_sequence in sequences_to_evaluate:
            evaluator = evaluation_class(user_id=user_id,
                                         model_name=f'{self.model_alias}_{self.train_alias}',
                                         predictor=self.model,
                                         item_dictionary=self.item_dictionary,
                                         session_id=session_id,
                                         encoded_sequence_to_evaluate=user_sequence,
                                         decoded_sequence=decoded_individual_sequence,
                                         auxiliary_sequence=auxiliary_item_sequence,
                                         max_sequence_length=self.max_sequence_size,
                                         vector_size=self.auxiliary_vector_size,
                                         top_k=self.top_k,
                                         mask_value=self.mask_value)

            evaluator.sequential_evaluation()
