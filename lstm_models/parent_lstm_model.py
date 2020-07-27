import datetime
import logging
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kerastuner import RandomSearch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('parent_lstm_model.py')
np.random.seed(42)


class BaseLSTM(object):

    def __init__(self, model_alias, train_alias, path_dictionary, training_dictionary, hyper_parameters):

        self.path_model_directory         : str = path_dictionary['path_model_directory']
        self.path_tuner_directory         : str = path_dictionary['path_tuner_directory']
        self.path_model_metadata          : str = path_dictionary['path_model_metadata']
        self.path_merged_dataframe        : str = path_dictionary['path_merged_dataframe']
        self.path_train_dataframe         : str = path_dictionary['path_train_dataframe']
        self.path_evaluation_dataframe    : str = path_dictionary['path_evaluation_dataframe']
        self.path_raw_dataframe           : str = path_dictionary['path_raw_dataframe']
        self.path_item2vec_model          : str = path_dictionary['path_item2vec_model']
        self.path_item_dictionary         : str = path_dictionary['path_item_dictionary']
        self.path_best_model              : str = path_dictionary['path_best_model']
        self.path_shared_folds            : str = path_dictionary['path_shared_folds']

        self.hyper_parameters = hyper_parameters

        self.random_state         : int = training_dictionary['random_state']
        self.k_split              : int = training_dictionary['k_split']
        self.max_trials           : int = training_dictionary['max_trials']
        self.executions_per_trial : int = training_dictionary['max_trials']
        self.epochs               : int = training_dictionary['epochs']
        self.batch_size           : int = training_dictionary['batch_size']
        self.evaluation_fraction  : float = training_dictionary['evaluation_fraction']
        self.max_sequence_size    : int = 10
        self.mask_value           : float = -9.0
        self.top_k                : int = 10
        self.thread_number        : int = 128

        self.number_of_distinct_items: int = 0
        self.encoding_vector_size: int = 0

        # LSTM Model
        self.model_alias: str = model_alias
        self.train_alias: str = train_alias
        self.model = None
        self.model_history = None

        self.item_dictionary = None
        self.user_sequences_dataframe = None
        self.evaluation_dataframe = None

        self.all_features = None
        self.all_targets  = None
        self.user_sequence_list  = None

        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None

        self.best_possible_models = []

    def log_event(self, message):
        current_time = datetime.datetime.now()
        logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

    def load_data(self):
        """
            Method will load the pre-processed data, split the evaluation part to another dataframe.
            Both dataframes then will be save to the project folder.
        """
        self.log_event('|----> Starting to load data..')

        # Load Mapped Item Dictionary
        items_file = open(self.path_item_dictionary, 'rb')
        self.item_dictionary = pickle.load(items_file)
        self.number_of_distinct_items = len(self.item_dictionary)

        # Load User Item Sequence with the catalog items
        self.user_sequences_dataframe = pd.read_pickle(self.path_merged_dataframe)

        # Shuffle the data, then split the evaluation data from the training data
        self.shuffle_user_sequence_data()

        evaluation_size = int(len(self.user_sequences_dataframe) * self.evaluation_fraction)

        self.evaluation_dataframe = self.user_sequences_dataframe.loc[np.random.choice(self.user_sequences_dataframe.index, size=evaluation_size, replace=False)]
        self.user_sequences_dataframe.drop(list(self.evaluation_dataframe.index.values), inplace=True)

        self.evaluation_dataframe.reset_index(inplace=True, drop=True)
        self.user_sequences_dataframe.reset_index(inplace=True, drop=True)

        # Convert necessary columns to list
        self.user_sequence_list = self.user_sequences_dataframe['item_sequence'].values

        self.save_data_to_disk(data_to_save=self.evaluation_dataframe, path_to_save=self.path_evaluation_dataframe)
        del self.evaluation_dataframe

        self.save_data_to_disk(data_to_save=self.user_sequences_dataframe, path_to_save=self.path_train_dataframe)

        self.log_event('|--------+ Memory Management: user_sequence_df has been removed from memory..')
        self.log_event('|----> Necessary data has been loaded..')

    def shuffle_user_sequence_data(self):
        """
            Method shuffles the user sequences data then resets its index
        """
        self.user_sequences_dataframe = shuffle(self.user_sequences_dataframe, random_state=self.random_state)
        self.user_sequences_dataframe.reset_index(inplace=True, drop=True)
        self.log_event('|----> Data is shuffled..')

    def create_common_folds_to_use(self):
        """
            Method is responsible of creating K-FOLD data.
            1. Initialize KFold() object with the necessary parameters.
            2. Create a loop which will iterate the total number of folds:
               2.1 Generate random indices for train and test split.
               2.2 Save the train test split data to the file system.

            Note that Folds are created on the shuffle data. In order to use the sequential order,
            set shuffle parameter to false when you are creating the KFOLD object. Same applies for
            the shuffle_user_sequence_data method which shuffles the sequence data.
        :rtype: object
        """
        folds = KFold(n_splits=self.k_split, random_state=self.random_state, shuffle=True)
        for split_count, (train_index, test_index) in enumerate(folds.split(self.user_sequences_dataframe)):
            self.save_data_to_disk(data_to_save=self.user_sequences_dataframe[self.user_sequences_dataframe.index.isin(train_index)],
                                   path_to_save=f'{self.path_shared_folds}/train_split_{str(split_count)}.fold')
            self.save_data_to_disk(data_to_save=self.user_sequences_dataframe[self.user_sequences_dataframe.index.isin(test_index)],
                                   path_to_save=f'{self.path_shared_folds}/test_split_{str(split_count)}.fold')

    def is_exist(self, directory_path_to_check):
        """
            Check if the given file exists
        @param directory_path_to_check: File to check
        @return: True if given file exists
        """
        return True if os.path.exists(directory_path_to_check) else False

    def create_meta_data_pickle(self):
        """
            Method create metadata fro the implemented model.
            1. Load the user sequence dataframe and the item dictionary
            2. Get the count of the distinct number of the items and the number of the total -disctinct- items
            3. After setting the class variables, retrieve the other necessary information by using
               retrieve_base_info_dictionary methods and save the dictionary to the file system.
        """
        self.user_sequences_dataframe = joblib.load(self.path_train_dataframe)

        self.item_dictionary = pickle.load(open(self.path_item_dictionary, 'rb'))
        self.number_of_distinct_items = len(self.item_dictionary)
        self.encoding_vector_size = len(self.item_dictionary)

        joblib.dump(value=self.retrieve_base_info_dictionary(), filename=self.path_model_metadata)

    def retrieve_base_info_dictionary(self):
        """
        @return: A dictionary of model metadata
        """
        return {
            'max_sequence_size'         : self.max_sequence_size,
            'number_of_distinct_items'  : self.number_of_distinct_items,
            'item_dictionary'           : self.item_dictionary,
            'encoding_vector_size'      : self.encoding_vector_size,
        }

    def load_model_metadata(self):
        """
            Load the dictionary contains metadata for the model and assign the variables.
        @return: Load model metadata from the file system.
        """
        info_dict = joblib.load(self.path_model_metadata)

        self.max_sequence_size = info_dict['max_sequence_size']
        self.number_of_distinct_items = info_dict['number_of_distinct_items']
        self.item_dictionary = info_dict['item_dictionary']
        self.encoding_vector_size = info_dict['encoding_vector_size']

        return info_dict

    def save_data_to_disk(self, data_to_save, path_to_save):
        """
            Method to use to save files to the file system.
        @param data_to_save: Source Data
        @param path_to_save: Directory path
        """
        if not os.path.exists(path_to_save):
            joblib.dump(value=data_to_save, filename=path_to_save)
            self.log_event(f'|--------> Saving Data: Data is saved under {path_to_save}..')
        else:
            self.log_event(f'|--------> Saving Data: {path_to_save} already exist in the project folder..')

    def create_features_and_targets(self):
        """
            Method takes the source data and creates features and labels for RNN.

            1. Create 2 numpy arrays to keep features and labels.
            2. Iterate through all the sequences:
                2.1 Add all the items except the last item in the sequence to the feature array.
                2.2 Add the last item in the sequence to the label array
        """
        self.log_event('|----> Creating feature and label list for user sequences..')

        self.all_features = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)
        self.all_targets = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype='int32')

        for index, sequence in enumerate(self.user_sequence_list):
            new_sequence = np.full((len(sequence) - 1), fill_value=self.mask_value, dtype='int32')
            for idx, item in enumerate(sequence[:-1]):
                new_sequence[idx] = item

            self.all_features[index] = new_sequence
            self.all_targets[index] = sequence[-1]

        del self.user_sequence_list
        self.log_event('|--------+ Memory Management: user_sequence_list has been removed from memory..')
        self.log_event('|----> Features and Targets are ready..')

    def encode_item(self, item_id):
        """
            Method encodes the given item by using One-Hot-Encoding.
        @param item_id: The objectID of the tour
        @return: The One-Hot-Encoded vector representation of the given tour object
        """
        vector = np.zeros(len(self.item_dictionary))
        encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(item_id)]
        vector[encoding] = 1
        return vector

    def scale_down_vector(self, vector_to_calc_l2):
        """
            Method calculates the l2-norm of the given vector and scales is by dividing the vector values to l2-norm.
        @param vector_to_calc_l2: Vector to scale
        @return:
        """
        from numpy.linalg import norm
        l2_norm = norm(vector_to_calc_l2)

        if l2_norm == 0:
            return vector_to_calc_l2

        return vector_to_calc_l2 / l2_norm

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

        self.all_features = masked_features
        del masked_features
        self.log_event('|----> Masking is completed..')

    def encode_targets(self):
        """
            Method iterates through all the labels and One-Hot-Encode the objectIDs
        """
        self.log_event('|----> One Hot Encoding the Targets..')

        count = 0
        targets = np.zeros(shape=(len(self.all_targets), self.encoding_vector_size), dtype='int8')

        for target in self.all_targets:
            encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(target)]
            vector = np.zeros(shape=self.encoding_vector_size, dtype='int8')
            vector[int(encoding)] = 1
            targets[count] = vector
            count += 1

        self.all_targets = targets
        del targets
        self.log_event('|----> One Hot Encoding has finished..')

    def start_hyper_parameter_tuning(self, split_number):
        """
            Method is responsible to find best RNN model fit the split data

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

        tuner.search(self.train_features, self.train_targets,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     verbose=2,
                     validation_data=(self.test_features, self.test_targets))

        # Get the trials
        trials = tuner.oracle.trials
        best_model = tuner.oracle.get_best_trials()[0].trial_id

        self.model = tuner.get_best_models(num_models=1)[0]

        self.model_history = self.model.fit(self.train_features, self.train_targets,
                                            epochs=self.epochs,
                                            batch_size=self.batch_size,
                                            verbose=2,
                                            validation_data=(self.test_features, self.test_targets))

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
            Model is responsible of building an RNN model with the decided parameters
        :param hp: Keras Tuner Parameters
        :return: Sequential Recurrent Neural Network Model
        """
        self.log_event('    -> Creating a model.')
        model = Sequential()

        model.add(self.add_mask_layer())

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
        model.add(Dense(self.number_of_distinct_items, activation=self.hyper_parameters['dense_activation']))

        # Compile the model
        opt = Adam(hp.Choice('learning_rate', values=self.hyper_parameters['learning_rate']))

        model.compile(loss=self.hyper_parameters['loss'], optimizer=opt, metrics=self.hyper_parameters['metric'])

        self.log_event('    -> Returning the model.')
        return model

    def add_mask_layer(self):
        """
            Masking Layer is one way to feed RNN with different size of input data. (https://keras.io/api/layers/core_layers/masking/)
        :return: Masking layer to be used in the proposed model
        """
        return Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.number_of_distinct_items))

    def load_fold_k_data_and_fit(self, split_number):
        """
            This method is only called if the Split Data is ready for model to be trained on and there is no best model saved in the file system.
            Consider a scenario that the split data is prepared and the program stopped. Instead of creating the same data, the function will find
            the data and load it into memory.

            After data is loaded, method will call start_hyper_parameter_tuning with the respective split number and start the hyper parameter tuning

        :param split_number: The number of the split data
        """
        self.train_features = joblib.load(self.path_model_directory + f'split_{split_number}/train_split_{split_number}_all_training_features.data')
        self.train_targets = joblib.load(self.path_model_directory + f'split_{split_number}/train_split_{split_number}_all_training_targets.data')

        self.test_features = joblib.load(self.path_model_directory + f'split_{split_number}/test_split_{split_number}_all_training_features.data')
        self.test_targets = joblib.load(self.path_model_directory + f'split_{split_number}/test_split_{split_number}_all_training_targets.data')

        self.start_hyper_parameter_tuning(split_number=str(split_number))

    def print_hyper_parameter_results(self, split_number, trials, best_model):
        """
            Method will prepare and save a TXT file which contains summary of the hyper parameter tuning. The best performed trial will be labelled as [BEST TRIAL]
        :param split_number: The number of the split data
        :param trials: Trial dictionary from Keras Tuner (tuner.oracle.trials)
        :param best_model: Trial ID of the best model
        """
        with open(f'{self.path_tuner_directory}/split_{split_number}/model_summary.txt', 'w') as text_file:
            self.model.summary(print_fn=lambda x: text_file.write(x + '\n'))
            text_file.write(f'\n')

            for trial in trials:
                if trial is best_model:
                    text_file.write(f'Trial ID: {trial} [BEST TRIAL] \n {("_" * 64)} \n\n')
                else:
                    text_file.write(f'Trial ID: {trial} \n {("_" * 64)} \n\n')

                text_file.write('\n   Selected Hyper Parameters:\n')

                for key, value in trials[trial].hyperparameters.values.items():
                    text_file.write(f'      |    {key}: {value} \n')

                text_file.write(f'{(" " * 4)}{("_" * 12)} \n')
                text_file.write(f'      | --> Trial Score: {str(trials[trial].score)} \n')
                text_file.write(f'{(" " * 4)}{("_" * 12)}\n\n')

    def plot_train_summary(self, plot_title, split_number):
        """
            Method plots the training results and save it to the respective trial folder
        :param plot_title: The Title of the plot
        :param split_number: The number of the split data
        """
        current_time = datetime.datetime.now()
        print('Model is successfully trained.. ' + str(current_time))
        print('Train on {} samples and Test on {} samples'.format(len(self.train_features), len(self.test_features)))

        title = 'Best Model ' + str(plot_title)
        loss_title = 'Best Model ' + str(plot_title) + ' Loss'
        plot_save_dir = f'{self.path_tuner_directory}split_{split_number}/{self.model_alias}_best_model_accuracy_{str(current_time.strftime("%Y-%m-%d_%H-%M-%S"))}.png'
        loss_plot_save_dir = f'{self.path_tuner_directory}split_{split_number}/{self.model_alias}_best_model_loss_{str(current_time.strftime("%Y-%m-%d_%H-%M-%S"))}.png'

        plt.plot(self.model_history.history[self.hyper_parameters['metric']])
        plt.plot(self.model_history.history[self.hyper_parameters['objective_function']])
        plt.title(title)
        plt.ylabel(self.hyper_parameters['metric'])
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(plot_save_dir, bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot training & validation loss values
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title(loss_title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(loss_plot_save_dir, bbox_inches='tight')
        plt.plot()
        plt.close()

    def retrieve_best_model(self, metric):
        """
            Method is responsible of finding the best model between all the splits. While doing that it also inserts the model performance to the database.
            In the end, best model is saved under the parent model folder with its train summary
        :param metric:
        """
        from utils.DatabaseHelper import insert_model_performance

        avg_val_accuracy_list = []
        for model in range(len(self.best_possible_models)):
            val_accuracy_list = self.best_possible_models[model][metric].values
            avg_accuracy = sum(val_accuracy_list) / len(val_accuracy_list)
            avg_val_accuracy_list.append(avg_accuracy)

        index = avg_val_accuracy_list.index(max(avg_val_accuracy_list))

        split = 0
        epoch = 1
        for row in self.best_possible_models[index].iterrows():
            insert_model_performance(model_name=f'{self.model_alias}_{self.train_alias}',
                                     split_number=split,
                                     epoch=epoch,
                                     train_loss=row[1][0],
                                     val_loss=row[1][2],
                                     train_accuracy=row[1][1],
                                     val_accuracy=row[1][3])
            epoch += 1
        split += 1

        model_path = f'{self.path_tuner_directory}split_{str(index)}'
        best_model_path = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".h5")][0]
        best_model_summary_path = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith("model_summary.txt")][0]

        self.model = load_model(best_model_path)

        # Save the best model to main folder
        self.model.save(f'{self.path_model_directory}{self.model_alias}_{self.train_alias}_best_model.h5')

        from shutil import copyfile
        copyfile(best_model_summary_path, f'{self.path_model_directory}{self.model_alias}_{self.train_alias}_model_summary.txt')

    def remove_catalogs(self, dataframe):
        """
            Method finds and removes the catalog visits from the user sequences.
            1. Iterate through dataframe.
            2. Append all the sequence objects except the last one to a new list:
                2.1 If list contains more than one item, append the last item to the list as well.
                ( The reason we are appending the last item after checking the size is that if there is no item in
                the list, there is no point adding the last item since there is no possible way to make predictions for
                the this item and validate the prediction with the ground truth. Because there is not ground truth..

        @param dataframe: Dataframe to remove catalog visits.
        @return: dataframe which contains user sequences with NO catalog visit
        """
        index_to_remove = []

        for index, row in dataframe.iterrows():
            count = 0
            items = []
            for item in row.item_sequence[:-1]:
                if item != 0:
                    items.append(item)
                count += 1
            if len(items) > 0:
                items.append(row.item_sequence[-1])
                dataframe.at[index, 'item_sequence'] = items
            else:
                index_to_remove.append(index)

        dataframe = dataframe[~dataframe.index.isin(index_to_remove)]
        dataframe.reset_index(inplace=True, drop=True)
        return dataframe

    def load_best_tuned_model(self, split_number):
        """
            This method only called if the training split is ready and there exists a model for the split data
            It will check the hyper parameter folder for the training result of the best split model, get its
            training results and append it to best_possible_models list
        :param split_number:
        """
        model_path = f'{self.path_tuner_directory}split_{str(split_number)}/'
        best_model_csv = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".csv")][0]
        best_model_df = pd.read_csv(best_model_csv)
        self.best_possible_models.append(best_model_df)

        self.log_event('Current folder has trained model in it. No need for training.')

    def load_evaluation_data(self):
        """
            Method is responsible of loading the necessary metadata and the evaluation data to test the best RNN model.
            It will drop the user_log_list from the evaluation dataframe since it is not needed for the evaluation scenario.
        :return:
        """
        self.load_model_metadata()

        evaluation_dataframe = joblib.load(self.path_evaluation_dataframe)

        # User log list column is not needed for evaluation thus can be dropped
        evaluation_dataframe.drop(['user_log_list'], axis=1, inplace=True)

        return evaluation_dataframe

    def evaluation_creator(self, sequences_to_evaluate, evaluation_class):
        """
            Method is responsible to apply evaluation on multi-threading.
            1. Calculate the portion of the evaluation data and assign as buckets.
            2. Each bucket will be given to a separate thread and perform evaluation.
        :param sequences_to_evaluate: List of sequences to evaluate
        :param evaluation_class: One of the evaluation classes located under model_evaluation folder. Model class needs to be sent as parameters since some models use
        different methods to evaluate sequences
        """
        import threading

        bucket_size = int(len(sequences_to_evaluate) / self.thread_number)
        lower_limit = 0
        upper_limit = bucket_size

        self.log_event(f'Sequence evaluation has started with {self.thread_number} Threads with {bucket_size} Buckets ..')

        threads = []
        for bucket in range(self.thread_number):
            individual_thread = threading.Thread(target=self.multi_thread_sequence_evaluation, args=(sequences_to_evaluate[lower_limit:upper_limit], evaluation_class))
            lower_limit = upper_limit
            upper_limit += bucket_size

            if bucket == self.thread_number - 2:
                upper_limit = len(sequences_to_evaluate)

            threads.append(individual_thread)
            individual_thread.start()

        for thread in threads:
            thread.join()

        self.log_event("Multi Thread Sequence evaluation has finished..")

    def multi_thread_sequence_evaluation(self, sequence_bucket, evaluation_class):
        """
            Method is responsible of creating an Evaluation Class Object and start the evaluation.
        :param sequence_bucket: The bucket contains the sequences to evaluate
        :param evaluation_class: Type of the evaluation class
        """
        for user_id, session_id, user_sequence, decoded_individual_sequence in sequence_bucket:
            evaluator = evaluation_class(user_id=user_id,
                                         model_name=f'{self.model_alias}_{self.train_alias}',
                                         predictor=self.model,
                                         item_dictionary=self.item_dictionary,
                                         session_id=session_id,
                                         encoded_sequence_to_evaluate=user_sequence,
                                         decoded_sequence=decoded_individual_sequence,
                                         max_sequence_length=self.max_sequence_size,
                                         vector_size=self.encoding_vector_size,
                                         top_k=self.top_k,
                                         mask_value=self.mask_value)

            evaluator.sequential_evaluation()