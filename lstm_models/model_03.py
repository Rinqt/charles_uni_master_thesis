from lstm_models.parent_lstm_model import *
from model_evaluation.SequenceEvaluationItem2Vec import Evaluator
import gensim

class LstmV3(BaseLSTM):
    def __init__(self, model_alias, train_alias, path_dictionary, training_dictionary, hyper_parameters):

        super().__init__(model_alias=model_alias,
                         train_alias=train_alias,
                         path_dictionary=path_dictionary,
                         training_dictionary=training_dictionary,
                         hyper_parameters=hyper_parameters)

        self.item2vec_model = None

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

            Note that there must be trained Item2Vec model for this class to function. If there is no Item2Vec model un-comment the line 36

        :return: Best RNN Model
        """

        # Train the i2v if it hasn't yet
        # self.train_i2v_model()
        self.load_i2v_model()

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
            self.encoding_vector_size = self.item2vec_model.vector_size

            # Iterate trough the split data for the training
            for split_number in range(self.k_split):
                split_path = f'split_{str(split_number)}/'
                split_directory = self.path_model_directory + split_path

                # Check the split directory is already created. If it is, then we can directly start the training by using the existing data
                if self.is_exist(split_directory):
                    try:
                        self.load_best_tuned_model(split_number)
                    except (IndexError, FileNotFoundError):
                        self.number_of_distinct_items = self.item2vec_model.vector_size
                        self.load_fold_k_data_and_fit(split_number=int(split_number))

                else:
                    # Create a folder for the split data and prepare the data for the training
                    os.makedirs(split_directory)

                    # Create an array which will contain train features-labels and test features-labels
                    train_array = np.full(4, fill_value=self.mask_value, dtype=object)
                    train_index = 0
                    for position, split_name in enumerate(['train_split_', 'test_split_']):
                        training_features_directory = split_directory + f'{split_name}{str(split_number)}_all_training_features.data'
                        training_targets_directory = split_directory + f'{split_name}{str(split_number)}_all_training_targets.data'
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

                    # Assign the input data to respective variables for the training
                    self.train_features = train_array[0]
                    self.train_targets = train_array[1]
                    self.test_features = train_array[2]
                    self.test_targets = train_array[3]
                    del train_array

                    self.number_of_distinct_items = self.encoding_vector_size
                    self.start_hyper_parameter_tuning(split_number)

                self.retrieve_best_model(metric=self.hyper_parameters['metric'])

    def process_training_data(self, fold_directory):
        """
            Method is responsible of creating an input data which is ready to be fit to RNN.
            1. Load the split data, remove the catalog information and transform the item_sequence of the dataframe to list
            2. Encode all the items as item2vec vectors.
            3. Create Features and Targets
            4. Apply Mask on the input features

            (Navigate to respective method for its own comments)

        :param fold_directory: The directory of the fold data
        """
        # Step 1: Remove the category pages from dataset. If user sequence contains only category pages, do not include it to training
        self.user_sequences_dataframe = joblib.load(fold_directory)
        self.user_sequences_dataframe = self.remove_catalogs(dataframe=self.user_sequences_dataframe)

        self.encoding_vector_size = self.item2vec_model.vector_size

        # Assign the processed item sequences
        self.user_sequence_list = self.user_sequences_dataframe['item_sequence'].values

        # Step 2: Remove the category pages from user sequences and encode all the items as item2vec vectors.
        self.encode_features_and_labels_to_i2v_vector()

        # Step 3: Create Features and Targets
        self.create_features_and_targets()

        # Step 4: Apply Mask to All Features
        self.apply_mask(all_features=self.all_features, number_of_features=self.encoding_vector_size)

    def encode_features_and_labels_to_i2v_vector(self):
        """
            Method will encode the features and the labels by using Item2Vec vectors.
            1. Iterate through self.user_sequence_list.
                1.1 Transform the visited objectIDs (except the catalog pages which has ID of zero) to their Item2Vec representations.
        """
        item2vec_sequences = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)

        for index, train_feature in enumerate(self.user_sequence_list):
            items = np.full((len(train_feature), self.encoding_vector_size), fill_value=self.mask_value, dtype='float32')
            for idx, item in enumerate(train_feature):
                i2v_representation = self.item2vec_model.wv.get_vector(str(item))
                items[idx] = i2v_representation
            item2vec_sequences[index] = items

        self.user_sequence_list = item2vec_sequences

    def create_features_and_targets(self):
        """
            Method takes the source data and creates features and labels for RNN.

            1. Create 2 numpy arrays to keep features and labels with the respective Item2Vec vector encoding size.
            2. Iterate through all the sequences:
                2.1 Add all the items except the last item in the sequence to the feature array.
                2.2 Add the last item in the sequence to the label array
        """
        self.log_event('|----> Creating feature and label list for user sequences..')

        self.all_features = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)
        self.all_targets = np.full((len(self.user_sequence_list), self.encoding_vector_size), fill_value=self.mask_value, dtype='float32')

        for index, sequence in enumerate(self.user_sequence_list):
            new_sequence = np.full((len(sequence) - 1, self.encoding_vector_size), fill_value=self.mask_value, dtype='float32')
            for idx, item in enumerate(sequence[:-1]):
                new_sequence[idx] = item

            self.all_features[index] = new_sequence
            self.all_targets[index] = sequence[-1]

        del self.user_sequence_list
        self.log_event('|--------+ Memory Management: user_sequence_list has been removed from memory..')
        self.log_event('|----> Features and Targets are ready..')

    def add_mask_layer(self):
        """
            Override the inherited method from the parent by setting the feature size to item2vec vector size
        :return: Masking layer for Model 3
        """
        return Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.encoding_vector_size))

    def create_meta_data_pickle(self):
        """
            Method is used to create necessary metadata information for the model.
            1. Load the sequence dataframe and item dictionary.
            2. Assign the number_of_distinct_items and encoding_vector_size with their values.

            Note that this model uses Item2Vec vectors for the input features.
            Thus, encoding_vector_size must be set to item2vec vector size (self.item2vec_model.vector_size)
        """
        self.user_sequences_dataframe = joblib.load(self.path_train_dataframe)

        self.item_dictionary = pickle.load(open(self.path_item_dictionary, 'rb'))
        self.number_of_distinct_items = len(self.item_dictionary)
        self.encoding_vector_size = self.item2vec_model.vector_size

        joblib.dump(value=self.retrieve_base_info_dictionary(), filename=self.path_model_metadata)

    def load_i2v_model(self):
        """
            Load the Item2Vec model to the memory
        """
        item2vec_model_path = self.path_item2vec_model
        file = open(item2vec_model_path, 'rb')
        self.item2vec_model = pickle.load(file)

    def train_i2v_model(self):
        """
            Method trains an Item2Vec model and saves it to the file system.
            1. Load the raw user sequenec data.
            2. Iterate through each sequence:
                2.1 Transform each sequence into a string of itemIDs (e.g. ['id1', 'id2', 'id3'] so that each item will be treated as words in a sentence).
                    Sequences should not contain catalog visits.
            3. Create and train and Item2Vec Model over the list of sequences.

        :return: Item2Vec model saved in the file system
        """
        self.user_sequences_dataframe = joblib.load(f'../source_data/merged_dataframe.pickle')
        self.user_sequence_list = self.user_sequences_dataframe['item_sequence'].values

        sequences = []
        for user_sequence in self.user_sequence_list:
            one_seq = []
            for item in user_sequence:
                if item != 0:
                    one_seq.append(str(item))
            sequences.append(one_seq)

        model = gensim.models.Word2Vec(sequences, min_count=0, size=128, window=1, iter=1024, workers=16)
        model.save("item2vec.model")

    def prepare_evaluation(self):
        """
            Method is responsible of creating an evaluation data suitable for the model.
            1. Check if the evaluation data exists. If yes, then call the evaluation class.
            2. If training data does not exist, create one:
                2.1 Load the evaluation dataframe from the source folder
                2.2 Remove the catalog information. (Because Model 3 is not able to handle catalog data)
                2.3 Create a new column to hold raw sequence information so that we can see which Item2vec vector refers which item.
                2.4 Create the suitable feature encodings
                2.5 Drop the unnecessary columns from the database and keep only user_id, session_id, item_sequence and decoded_item_sequence columns.
                2.6 Transform the columns to a list and then save it to the model folder for future trainings
                2.7 Load the model metadata and call the evaluation class.

            Note that item2vec model is also needs to be loaded before creating the evaluation data. This model uses Item2Vec vectors to encode the visited objects.

        """
        self.model = load_model(self.path_best_model)
        self.load_i2v_model()

        if not self.is_exist(f'{self.path_model_directory}evaluation_ready_sequences.data'):
            self.evaluation_dataframe = self.remove_catalogs(dataframe=self.load_evaluation_data())

            self.evaluation_dataframe['decoded_item_sequence'] = self.evaluation_dataframe['item_sequence'].copy(deep=True)
            self.evaluation_dataframe.drop(['session_start_time', 'catalog_item_list', 'good_catalog_items', 'sequence_length'], axis=1, inplace=True)

            self.user_sequence_list = self.evaluation_dataframe['item_sequence'].values
            self.encode_features_and_labels_to_i2v_vector()

            for index, sequence in enumerate(self.user_sequence_list):
                self.evaluation_dataframe.at[index, 'item_sequence'] = np.array(sequence)

            self.evaluation_dataframe['item_sequence'] = self.user_sequence_list

            sequences = self.evaluation_dataframe.values.tolist()
            joblib.dump(sequences, f'{self.path_model_directory}evaluation_ready_sequences.data')
        else:
            sequences = joblib.load(f'{self.path_model_directory}evaluation_ready_sequences.data')
            self.load_model_metadata()

        self.evaluation_creator(sequences_to_evaluate=sequences, evaluation_class=Evaluator)