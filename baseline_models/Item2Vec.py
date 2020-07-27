import datetime
import logging
import os
import pickle
from math import log

import gensim
import joblib
from pandas import np

from utils.DatabaseHelper import insert_evaluation
from utils.performance_metrics.Metrics import recall, precision, mrr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Item2Vec.py')

def log_event(message):
    current_time = datetime.datetime.now()
    logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

def train_item2vec():
    """
        Model checks if there is already trained Item2Vec model.
        1. If not load the source dataframe.
        2. Remove the catalog information from the sequences.
        3. Create an Item2Vec object and train it over the sequences. Note that sequences must
           be string of objectIds like words in a sentence. ['objID1', 'objID2', ...]
    """
    if not os.path.exists('basic_item2vec.model'):
        user_sequences_dataframe = joblib.load(f'../source_data/merged_dataframe.pickle')
        user_sequences_dataframe = remove_catalogs(dataframe=user_sequences_dataframe)

        sequences = user_sequences_dataframe['item_sequence'].values

        model = gensim.models.Word2Vec(sequences, min_count=0, size=128, window=1, iter=1024, workers=16)
        model.save("basic_item2vec.model")
    else:
        log_event('|----> Item2Vec Model is found no need for training..')


def load_i2v_model(item2vec_model_path):
    file = open(item2vec_model_path, 'rb')
    log_event('|----> Item2Vec Model is load from file..')
    return pickle.load(file)

def remove_catalogs(dataframe):
    """
        Method finds and removes the catalog visits from the user sequences.
        1. Iterate through dataframe.
        2. Append all the sequence objects except the last one to a new list:
            2.1 If list containes more than one item, append the last item to the list as well.
            ( The reason we are appending the last item after checking the size is that if there is no item in
            the list, there is no point adding the last item since it will bu used as label for the training
            and we cannot use the sequence with 1 visited item to train the RNN.

    @param dataframe: Dataframe to remove catalog visits.
    @return: dataframe which contains user sequences with NO catalog visit
    """
    index_to_remove = []

    for index, row in dataframe.iterrows():
        count = 0
        items = []
        for item in row.item_sequence[:-1]:
            if item != 0:
                items.append(str(item))
            count += 1
        if len(items) > 0:
            items.append(str(row.item_sequence[-1]))
            dataframe.at[index, 'item_sequence'] = items
        else:
            index_to_remove.append(index)

    dataframe = dataframe[~dataframe.index.isin(index_to_remove)]
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe

def evaluate_model():
    """
        Method to start evaluation
        1. Check if the evaluation data is ready. If yes evaluate the model. If not create the data.
        2. If evaluation data is not ready:
            2.1 Load the source evaluation dataframe
            2.2 Remove catalogs from the sequences.
            2.3 Create a copy of raw sequence information to a new column. (this is useful for the models
                use encodings)
            2.4 Remove the columns other than user_id, session_id, user_sequence_ decoded_individual_sequence
                columns.
            2.5 Convert the data to a list and save it to file system.
    """
    item2vec_model = load_i2v_model(f'../utils/item2vec.model')

    if not os.path.exists('evaluation_data/evaluation_sequences.pickle'):
        evaluation_dataframe = joblib.load(f'../source_data/common_data/evaluation_dataframe.pickle')
        evaluation_dataframe.drop(['user_log_list'], axis=1, inplace=True)
        evaluation_dataframe = remove_catalogs(dataframe=evaluation_dataframe)

        evaluation_dataframe['decoded_item_sequence'] = evaluation_dataframe['item_sequence'].copy(deep=True)
        evaluation_dataframe.drop(['catalog_item_list', 'session_start_time', 'good_catalog_items'], axis=1, inplace=True)

        sequences = evaluation_dataframe.values.tolist()
        sequences = np.array(sequences)
        joblib.dump(sequences, 'evaluation_data/evaluation_sequences.pickle')
    else:
        sequences = joblib.load('evaluation_data/evaluation_sequences.pickle')

    evaluation_creator(model=item2vec_model,
                       model_name='base_item2vec_model',
                       sequence_to_evaluate=sequences)

def evaluation_creator(model, model_name, sequence_to_evaluate):
    log_event('|-- Evaluation started..')

    items_file = open(f'../source_data/item.dictionary', 'rb')
    item_dictionary = pickle.load(items_file)

    for user_id, session_id, user_sequence, decoded_individual_sequence in sequence_to_evaluate:
        evaluator = Evaluator(user_id=user_id,
                              model_name=model_name,
                              predictor=model,
                              item_dictionary=item_dictionary,
                              session_id=session_id,
                              encoded_sequence_to_evaluate=np.array(user_sequence),
                              decoded_sequence=decoded_individual_sequence,
                              top_k=10)

        evaluator.sequential_evaluation()

    log_event('|-- Evaluation finished..')

class Evaluator(object):
    def __init__(self, user_id, model_name, predictor, item_dictionary, session_id, encoded_sequence_to_evaluate, decoded_sequence, top_k):
        self.user_id = user_id
        self.model_name = model_name
        self.predictor = predictor
        self.item_dictionary = item_dictionary
        self.session_id = session_id
        self.encoded_sequence_to_evaluate = encoded_sequence_to_evaluate
        self.decoded_sequence = decoded_sequence
        self.top_k = top_k
        self.increment_by = 1
        self.ndcg = 0.
        self.real = 0.
        self.real_index = 0
        self.metrics = {'precision': precision, 'recall': recall, 'mrr': mrr}
        self.prm = np.zeros(len(self.metrics.values()))
        self.recommendation = None
        self.is_trivial_prediction = False
        self.is_only_catalog_prediction = False

    def sequential_evaluation(self):


        for index in range(len(self.encoded_sequence_to_evaluate) - 1):
            self.is_trivial_prediction = False
            self.is_only_catalog_prediction = False

            self.prm = np.zeros(len(self.metrics.values()))
            self.ndcg = 0

            clicked_item = self.encoded_sequence_to_evaluate[index]
            self.evaluate_sequence(clicked_item, self.decoded_sequence)

            insert_evaluation(user_id=self.user_id,
                              session_id=self.session_id,
                              precision=self.prm[0],
                              recall=self.prm[1],
                              mrr=self.prm[2],
                              ndcg=self.ndcg,
                              predictor_name=self.model_name,
                              trivial_prediction=self.is_trivial_prediction,
                              catalog_prediction=self.is_only_catalog_prediction,
                              catalog_count=0,
                              ground_truth=self.real,
                              sequence=' '.join(map(str, self.decoded_sequence)),
                              input_sequence=int(clicked_item),
                              input_sequence_length=1,
                              user_sequence_length=len(self.decoded_sequence),
                              predictions=' '.join(map(str, self.recommendation)))

            self.increment_by += 1

            if self.increment_by == len(self.decoded_sequence):
                break


    def evaluate_sequence(self, sequence, gt):
        user_profile = sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.item2vec_calculate_similarity(item_id=user_profile, item_list=self.item_dictionary)[:self.top_k]

        self.real = ground_truth[0]
        self.real_index = gt.index(self.real)

        # Calculate Precision, Recall and MMR
        for i, metric_function in enumerate(self.metrics.values()):
            self.prm[i] += metric_function([self.real], self.recommendation)
        # Calculate nDCG
        if 0.0 not in self.prm:
            # Then calculate nDCG
            self.calculate_ndcg()


    def item2vec_calculate_similarity(self, item_id, item_list):
        """
        :summary:  Creates a recommendation list that contains ids of the item that is similar to given id.
        :workflow:
                    -> Load the item2vec model from database.
                    -> Use model to find similar items to given item id.
                    -> Create a recommendation list and start appending predicted item ids (Note that some item ids
                       might not be in the database)
                    -> Return the list.

        :return: recommended_items_set: List of item ids to use for recommendation
        """
        try:
            similar_items = self.predictor.wv.most_similar(str(item_id), topn=20)
        except Exception as err:
            logging.error('Item2Vec Algorithm could not find the given item in the vocabulary', '\n', err)
            return [-1]

        recommended_items_set = []
        for rec_item in similar_items:
            encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(int(rec_item[0]))]
            if encoding in item_list:
                recommended_items_set.append(int(rec_item[0]))


        return recommended_items_set[:10]

    def calculate_ndcg(self):
        one_zero_list = []
        for pred in self.recommendation:
            if pred == self.real:
                one_zero_list.append(1)
            else:
                one_zero_list.append(0)

        pos = 1
        if 1 in one_zero_list:
            if self.real in self.decoded_sequence[self.real_index + 1:]:
                self.ndcg = -1
                self.is_trivial_prediction = True
                return self.ndcg

            dcg_score = 0
            for item in one_zero_list:
                if item != 0:
                    dcg_score += 1 / log(pos+1, 2)
                pos += 1

            self.ndcg += dcg_score

train_item2vec()
evaluate_model()