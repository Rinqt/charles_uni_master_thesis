import json
import glob

from lstm_models.model_01 import LstmV1
from lstm_models.model_02 import LstmV2
from lstm_models.model_03 import LstmV3
from lstm_models.model_04 import LstmV4
from lstm_models.model_05 import LstmV5
from lstm_models.model_06 import LstmV6
from lstm_models.model_07 import LstmV7

from utils.DatabaseHelper import insert_model_data


def create_path_information(model_alias:str, model_train_type:str):
    """

    :param model_alias:
    :param model_train_type:
    """
    lstm_model_parameters = \
        {
            'path_model_directory'      : f'model_data/{model_alias}/',
            'path_tuner_directory'      : f'model_data/{model_alias}/hpt/{model_train_type}/',
            'path_model_metadata'       : f'model_data/{model_alias}/{model_alias}.metadata',
            'path_best_model'           : f'model_data/{model_alias}/{model_alias}_{model_train_type}_best_model.h5',
            'path_raw_dataframe'        : f'model_data/preprocessed_dataframe.pickle',
            'path_item2vec_model'       : f'utils/item2vec.model',
            'path_item_dictionary'      : f'source_data/item.dictionary',
            'path_shared_folds'         : f'source_data/common_data/folds/',
            'path_merged_dataframe'     : f'source_data/merged_dataframe.pickle',
            'path_train_dataframe'      : f'source_data/common_data/train_dataframe.pickle',
            'path_evaluation_dataframe' : f'source_data/common_data/evaluation_dataframe.pickle',
        }
    return lstm_model_parameters


# Define Hyper Parameters for models that will be built
hyper_parameters = \
    {
        'objective_function'    : 'val_accuracy',
        'lstm_units'            : {'min': 32, 'max': 128, 'step': 32},
        'dropout'               : [0.1, 0.3, 0.5],
        'lstm_layer_activation' : 'relu',
        'lstm_layer_dropout'    : 0.1,
        'dense_activation'      : 'softmax',
        'learning_rate'         : [0.01, 0.001, 0.0001],
        'metric'                : 'accuracy',
        'loss'                  : 'categorical_crossentropy',
    }


# Define your models below. Structure: {'model_alias':{'model_class_name':class_object, 'model_description':'<insert_model_description>'}}
lstm_models = \
    {

        'model_01': {'class_name': LstmV1,
                            'description': 'Features: Items in the sequence except the last one - itemIDs kept as integers. '
                                           'Labels: Last Item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using integer itemIDs. It is not able to use information from category pages, '
                                           'thus cannot make predictions for category pages.'},

        'model_02': {'class_name': LstmV2,
                            'description': 'Features: Items in the sequence except the last one - One Hot Encoded. '
                                           'Labels: Last item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using the OHE vectors. It is not able to use information from category pages, '
                                           'thus cannot make predictions for category pages.'},

        'model_03': {'class_name': LstmV3,
                            'description': 'Features: Items in the sequence except the last one - itemIDs are converted to i2v representation. '
                                           'Labels: Last item in the sequence - i2v representation. '
                                           'Rnn is fit by using the i2v vectors. It is not able to use information from category pages, '
                                           'thus cannot make predictions for category pages.'},

        'model_04': {'class_name': LstmV4,
                            'description': 'Features: Items in the sequence except the last one - itemIDs are converted to i2v representation. '
                                           'Labels: Last item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using the i2v features. It is not able to use information from category pages, '
                                           'thus cannot make predictions for category pages.'},

        'model_05': {'class_name': LstmV5,
                            'description': 'Features: Items in the sequence except the last one AND category pages. '
                                           '   Items in the category pages are identified by parsing the logs and scaled based on visibility and click action.'
                                           '   All the items in the category page inserted into one vector and scaled down by using l2 norm. '
                                           'Labels: Last item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using the items that users visited and items in the category pages. This model is able to make '
                                           'recommendation for category pages.'},

        'model_06': {'class_name': LstmV6,
                            'description': 'Features: Items in the sequence except the last one AND category pages. '
                                           '  AUXILIARY INPUT: Tour meta data as used as second branch of the RNN. '
                                           '  Items in the category pages are identified by parsing the logs and scaled based on visibility and click action. '
                                           '  All the items in the category page inserted into one vector and scaled down by using l2 norm. '
                                           'Labels: Last item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using the items that users visited and items in the category pages with additional object metadata. '
                                           'This model is able to make recommendation for category pages.'},
        'model_07': {'class_name': LstmV7,
                            'description': 'Features: Only item metadata. '
                                           'Labels: Last item in the sequence - One Hot Encoded. '
                                           'Rnn is fit by using the items that users visited and items in the category pages with additional object metadata. '},


    }

# Define your training options below
training_options = \
    {
        '8_epoch_512_batch_test'  : {'test_fraction': 0.25, 'evaluation_fraction': 0.25, 'max_trials': 3, 'execution_per_trial': 3, 'epochs': 8, 'batch_size': 512, 'k_split': 4,
                                'random_state': 42},
    }

def train_evaluate_models():
    for model_name, model_metadata in lstm_models.items():

        model_class = model_metadata['class_name']
        model_description = model_metadata['description']

        for option_alias, train_options in training_options.items():

            if 'model_03' in model_name:
                hyper_parameters['objective_function'] = 'val_mean_absolute_error'
                hyper_parameters['metric'] = 'mean_absolute_error'
                hyper_parameters['loss'] = 'mean_absolute_error'
            else:
                hyper_parameters['objective_function'] = 'val_accuracy'
                hyper_parameters['dense_activation'] = 'tanh'
                hyper_parameters['metric'] = 'accuracy'
                hyper_parameters['loss'] = 'categorical_crossentropy'

            processed_paths = create_path_information(model_alias=model_name, model_train_type=option_alias)
            lstm_model = model_class(model_alias=model_name,
                                     train_alias=option_alias,
                                     path_dictionary=processed_paths,
                                     training_dictionary=train_options,
                                     hyper_parameters=hyper_parameters)

            # insert model params to db
            insert_model_data(model_type=model_name,
                              training_type=option_alias,
                              description=model_description,
                              hyper_parameters=json.dumps(hyper_parameters),
                              meta_data=json.dumps(training_options))

            is_data_ready = True if (lstm_model.k_split * 2) == fold_counter else False
            if not is_data_ready:
                lstm_model.load_data()
                lstm_model.create_common_folds_to_use()

            lstm_model.train_model()
            lstm_model.prepare_evaluation()


path_to_folds = f'source_data/common_data/folds'
fold_counter = len(glob.glob1(path_to_folds, "*.fold"))
train_evaluate_models()