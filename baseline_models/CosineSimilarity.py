import datetime
import os

import gensim
import logging
import pickle
import re

import joblib
import pandas as pd
from pandas import np
from math import log

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from utils.performance_metrics.Metrics import recall, precision, mrr
from utils.DatabaseHelper import insert_evaluation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BaseModelEvaluation.py')


class BasicCosSimilarity(object):

    def log_event(self, message):
        current_time = datetime.datetime.now()
        logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

    def __init__(self):
        self.path_content_dataframe = f'D:/Thesis/recsys_thesis/data/auxiliary_df_raw.pickle'
        self.content_dataframe = None

        self.cosine_similarity_matrix = None
        self.tour_ids = None
        self.characters_to_remove = '\ |\[|\]|\!|\/|\;|\:'
        self.count_vectorizer = None
        self.count_matrix = None
        self.recommended_tours = []

        # Evaluation
        self.evaluation_dataframe = None
        self.sequences_to_evaluate = None

        self.trivial_prediction = False
        self.trivial_prediction = False

    def start_evaluation(self):
        """
            Main methods to work with Cosine Similary. Method will first check if the necessary (object dictionary and dataframe) then evaluate the model.
            If files do not exist, it will create them first, save into the file system then evaluate the model.

        """
        if os.path.exists('basic_cos_sim_objects.dictionary'):
            self.load_metadata()
        else:
            if not os.path.exists('cosine_similarity_source.dataframe'):
                self.load_metadata_dictionary()
                self.prepare_dataframe(drop_duplicate_tours=True)
            else:
                self.content_dataframe = joblib.load('cosine_similarity_source.dataframe')
            self.create_cosine_similarity_vector()
            self.save_objects()

        self.evaluate_model()

    def start(self):
        if os.path.exists('basic_cos_sim_objects.dictionary'):
            self.load_metadata()
        else:
            if not os.path.exists('cosine_similarity_source.dataframe'):
                self.load_metadata()
                self.prepare_dataframe(drop_duplicate_tours=True)
            else:
                self.content_dataframe = joblib.load('cosine_similarity_source.dataframe')
            self.create_cosine_similarity_vector()
            self.save_objects()

        self.evaluate_model()

    def renew_title(self, row):
        """
            Method gets a dataframe row, process the title of the tour object by "item_id;valid_from" structure
            then returns it.
        @param row: Rows of a dataframe
        @return:
        """
        new_title = str(row.item_id) + ';' + str(row.valid_from)
        return re.sub('-', '', new_title)

    def load_metadata(self):
        basic_cos_sim_dict = joblib.load('basic_cos_sim_objects.dictionary')

        self.content_dataframe = basic_cos_sim_dict['content_dataframe']
        self.cosine_similarity_matrix = basic_cos_sim_dict['sim_matrix']
        self.tour_ids = basic_cos_sim_dict['tour_ids']
        self.count_vectorizer = basic_cos_sim_dict['count_vectorizer']
        self.count_matrix = basic_cos_sim_dict['count_matrix']

        self.log_event('|-- Dataframe load from the disk..')

    def concat_vectors(self, row):
        """
            Method get a dataframe row that have vectors of information related to a tour object.
            Then it merges the all vectors to a single one.
        @param row: Row that contains all the vector information about the tour objecy
        @return: merged vector
        """
        words = ''
        for vector in row[1:]:
            words = words + ''.join(str(vector)) + ''
        processed_word = re.sub(self.characters_to_remove, '', words)
        return processed_word

    def prepare_dataframe(self, drop_duplicate_tours):
        """
            Method creates a dataframe that can be used to built similarity matrix.
            1. Drop the duplicate values based on the given parameter.
            2. Replace the item id by using the renew_title method
            3. Set dataframe index to the newly created titles and remove columns which do not contain vector
            4. Concatenate vectors
            5. Drop all the other columns expect the merged vector column
            6. Save the file to the file system.

        @param drop_duplicate_tours: Boolean to drop duplicated tour rows.
        """
        self.log_event('|----> Pre-Processing Started..')

        if drop_duplicate_tours:
            self.content_dataframe.drop_duplicates(subset="item_id", keep='first', inplace=True)

        self.content_dataframe['item_id'] = self.content_dataframe.apply(self.renew_title, axis=1)
        self.content_dataframe.set_index('item_id', inplace=True)

        self.content_dataframe.drop(['tour_id', 'to_date', 'valid_from', 'valid_to'], axis=1, inplace=True)

        self.content_dataframe['item_vector'] = self.content_dataframe.apply(self.concat_vectors, axis=1)
        self.content_dataframe.drop(columns=[col for col in self.content_dataframe.columns if col != 'item_vector'], inplace=True)

        joblib.dump(self.content_dataframe, f'cosine_similarity_source.dataframe')

        self.log_event('|----> Pre-Processing Finished..')


    def create_cosine_similarity_vector(self):
        """
            Method creates Cosine Similarity for the tour objects.
            1. Create a CountVectorizer object from sklearn
            2. Fit the vectorizer on the item_vector data
            3. Save the tour_ids (which are the index of the Dataframe into a list to use them later to find the
               similar objects.
            4. Create a cosine similarity by using the cosine_similarity method from sklearn.
        """
        self.log_event('|----> Creating Cosine Similarity Matrix..')

        self.count_vectorizer = CountVectorizer()
        self.count_matrix = self.count_vectorizer.fit_transform(self.content_dataframe['item_vector'])

        self.tour_ids = pd.Series(self.content_dataframe.index)

        self.cosine_similarity_matrix = cosine_similarity(self.count_matrix, self.count_matrix)

        self.log_event('|----> Cosine Similarity Matrix Created..')

    def remove_catalogs(self, dataframe):
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

    def save_objects(self):
        """
             Method save necessary information to a dictionary then saves the dictionary to the file system.
         @return:
        """
        # Save necessary objects to pickle
        dict_to_save = {'content_dataframe': self.content_dataframe,
                        'sim_matrix': self.cosine_similarity_matrix,
                        'tour_ids': self.tour_ids,
                        'count_vectorizer': self.count_vectorizer,
                        'count_matrix': self.count_matrix}

        joblib.dump(dict_to_save, 'cosine_similarity_objects.dictionary')
        self.log_event('|-- Objects saved into disk..')

    def load_metadata_dictionary(self):
        self.content_dataframe = pd.read_pickle(self.path_content_dataframe)
        self.log_event('|-- Dataframe load from the disk..')

    def evaluate_model(self):
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
        if not os.path.exists('evaluation_data/evaluation_sequences.pickle'):
            self.evaluation_dataframe = joblib.load(f'../source_data/common_data/evaluation_dataframe.pickle')
            self.evaluation_dataframe.drop(['user_log_list'], axis=1, inplace=True)
            self.evaluation_dataframe = self.remove_catalogs(dataframe=self.evaluation_dataframe)

            self.evaluation_dataframe['decoded_item_sequence'] = self.evaluation_dataframe['item_sequence'].copy(deep=True)

            self.evaluation_dataframe.drop(['catalog_item_list', 'session_start_time', 'good_catalog_items', 'sequence_length'], axis=1, inplace=True)

            self.sequences_to_evaluate = self.evaluation_dataframe.values.tolist()
            self.sequences_to_evaluate = np.array(self.sequences_to_evaluate)
            joblib.dump(self.sequences_to_evaluate, 'evaluation_data/evaluation_sequences.pickle')
        else:
            self.sequences_to_evaluate = joblib.load('evaluation_data/evaluation_sequences.pickle')

        self.evaluation_creator(model_name='cosine_similarity')


    def evaluation_creator(self, model_name):
        """
            Method iterates the user sequences, create an Evaluator Class Object and evaluates the sequences.
        @param model_name: The name of the model used for the evaluation
        """
        self.log_event('|-- Evaluation started..')

        for user_id, session_id, user_sequence, decoded_individual_sequence in self.sequences_to_evaluate:
            evaluator = Evaluator(user_id=user_id,
                                  model_name=model_name,
                                  session_id=session_id,
                                  encoded_sequence_to_evaluate=np.array(user_sequence),
                                  decoded_sequence=decoded_individual_sequence,
                                  top_k=10,
                                  increment_by=1,
                                  content_dataframe=self.content_dataframe,
                                  cosine_similarity_matrix=self.cosine_similarity_matrix,
                                  tour_ids=self.tour_ids)

            evaluator.sequential_evaluation()

        self.log_event('|-- Evaluation finished..')



class Evaluator(object):
    def __init__(self, user_id, model_name, session_id, encoded_sequence_to_evaluate, decoded_sequence, top_k, increment_by, content_dataframe, cosine_similarity_matrix, tour_ids):
        self.user_id = user_id
        self.model_name = model_name
        self.session_id = session_id
        self.encoded_sequence_to_evaluate = encoded_sequence_to_evaluate
        self.decoded_sequence = decoded_sequence
        self.top_k = top_k
        self.increment_by = increment_by
        self.ndcg = 0.
        self.real = 0.
        self.real_index = 0
        self.metrics = {'precision': precision, 'recall': recall, 'mrr': mrr}
        self.prm = np.zeros(len(self.metrics.values()))
        self.recommendation = None
        self.content_dataframe = content_dataframe
        self.cosine_similarity_matrix = cosine_similarity_matrix
        self.tour_ids = tour_ids
        self.is_trivial_prediction = False
        self.is_only_catalog_prediction = False

    def sequential_evaluation(self):
        """
            Method iterates through the user sequences (not including the last item) and evaluates the next-item
            predictions for the given item visit.
            1. Get the clicked item
            2. Call the evaluate_sequence method by giving the clicked item and the sequence information to get
               recommendations and its evaluation.
            3. Insert the evaluation data to the database
        """
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
        """
            Method recevies the current user profile (clicked item(s)) and creates next-item recommendations.
            Generated recommendations are compared with the ground truth and its performance is evaluated.
        @param sequence: Visited object ID or its encoding (based on a used model)
        @param gt: The rest of the sequence after the clicked_item objectID.
        """
        user_profile = sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.find_similar_tours(tour_id=user_profile)[:self.top_k]

        if not len(self.recommendation) < 1:

            self.real = ground_truth[0]
            self.real_index = gt.index(self.real)

            # Calculate Precision, Recall and MMR
            for i, metric_function in enumerate(self.metrics.values()):
                self.prm[i] += metric_function([self.real], self.recommendation)
            # Calculate nDCG
            if 0.0 not in self.prm:
                # Then calculate nDCG
                self.calculate_ndcg()

    def find_similar_tours(self, tour_id):
        """
            Methods receives the tour id and finds the similar tours.

            1. Iterate through the tour_ids list and compare the given tour id with the ones that were processed.
            2. If tour id is found, then get its location in the tour_ids list and use the location to lacote the
               similarity socres on the cosine similarity matrix.
        @param tour_id: ID of the tour object that we want to create recommendations.
        @return: Top-10 Recommendation
        """
        recommended_tours = []
        tour_processed_id = -1

        for index, processed_tour_id in enumerate(self.tour_ids):
            if tour_id == int(processed_tour_id.split(';')[0]):
                tour_processed_id = processed_tour_id
                break

        if tour_processed_id == -1:
            return []
        # Get the real tour id matches with processed tour id
        tour_pos = self.tour_ids[self.tour_ids == tour_processed_id].index[0]

        # Create a Series with the similarity scores (in ASC order)
        score_series = pd.Series(self.cosine_similarity_matrix[tour_pos]).sort_values(ascending=False)

        # Retrieve the top ten similar tours. Note that the first tour will be always the same tour that we are looking similar tours
        top_10_indexes = list(score_series.iloc[1: self.top_k + 1].index)

        for i in top_10_indexes:
            recommended_tours.append(list(self.content_dataframe.index)[i].split(';')[0])

        return list((map(int, recommended_tours))) # convert string ids to int

    def calculate_ndcg(self):
        """
            Calculate the nDCG score. If the ground truth item is located in the sequence marked it as trivial prediction
        """
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

# BasicCosSimilarity
cos_sim_1 = BasicCosSimilarity()
cos_sim_1.start_evaluation()