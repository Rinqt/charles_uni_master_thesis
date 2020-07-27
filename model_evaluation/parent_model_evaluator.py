from utils.performance_metrics.Metrics import recall, precision, mrr
from math import log

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('parent_model_evaluator.py')

class BaseEvaluator(object):

    def __init__(self, user_id, model_name, predictor, item_dictionary, session_id, encoded_sequence_to_evaluate, decoded_sequence, max_sequence_length, vector_size, top_k, mask_value):
        self.user_id = user_id
        self.model_name = model_name
        self.predictor = predictor
        self.item_dictionary = item_dictionary
        self.session_id = session_id
        self.encoded_sequence_to_evaluate = encoded_sequence_to_evaluate
        self.decoded_sequence = decoded_sequence
        self.top_k = top_k
        self.increment_by = 1
        self.max_sequence_length = max_sequence_length
        self.vector_size = vector_size
        self.mask_value = mask_value
        self.ndcg = 0.
        self.real = 0.
        self.real_index = 0
        self.metrics = {'precision': precision, 'recall': recall, 'mrr': mrr}
        self.prm = np.zeros(len(self.metrics.values()))

        self.seq_len = None
        self.rnn_input = None
        self.recommendation = None
        self.is_trivial_prediction = None
        self.is_only_catalog_prediction = None
        self.catalog_counter = None

    def evaluate_sequence(self, sequence, gt):
        """
            Method wil assign the user_profile and ground_truth based on the user sequence.
            User profile will be sent to the RNN for the predictions
            Then method will calculate precision, recall and MRR first. If prediction has performance score
            bigger than zero, we will calculate the nDCG Score.
        :param sequence: User sequence to evaluate
        :param gt: Ground Truth
        """
        user_profile = sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.lstm_calculate_similarity(user_sequence=user_profile)[:self.top_k]

        self.real = ground_truth[0]
        self.real_index = gt.index(self.real)

        # Calculate Precision, Recall and MMR
        for i, metric_function in enumerate(self.metrics.values()):
            self.prm[i] += metric_function([self.real], self.recommendation)

        if 0.0 not in self.prm:
            # Then calculate nDCG
            self.calculate_ndcg()

    def lstm_calculate_similarity(self, user_sequence):
        """
            Get the Top-K Recommendation by using the user profile.
            Note that recommended items will be item-dictionary encoding. So they need to be
            transformed to their original objectIDs.
        :param user_sequence: User profile which contains the encoded implicit feedback
        :return: Top-K Recommendation List
        """
        try:
            similar_items = self.predictor.predict(user_sequence)
        except Exception as err:
            logging.error('LSTM Algorithm could not find the given item in the vocabulary', '\n', err)
            return [-1]

        top_k = (-similar_items).argsort()[0][:self.top_k]

        # Get the original objectID from the item-dictionary
        recommended_items_set = []
        for rec_item in top_k:
            if rec_item in self.item_dictionary:
                encoded_item = self.item_dictionary.get(rec_item)
                recommended_items_set.append(encoded_item)

        return recommended_items_set[:self.top_k]

    def calculate_ndcg(self):
        """
            Calculate the nDCG Score of the recommendation
            1. Create a binary list which has exactly same size with self.recommendation
            2. Fill the array with zeros, except the location for the prediction. Put 1 for the prediction location
            3. Iterate through the binary list and calculate the nDCG Score based of the prediction location
                3.1 If ground truth is located later in the sequence mark the prediction trivial by setting self.is_trivial_prediction = true
        """
        one_zero_list = []
        for pred in self.recommendation:
            if pred == self.real:
                one_zero_list.append(1)
            else:
                one_zero_list.append(0)

        pos = 1
        if 1 in one_zero_list:
            dcg_score = 0
            for item in one_zero_list:
                if item != 0:
                    dcg_score += 1 / log(pos+1, 2)
                pos += 1

            self.ndcg += dcg_score

            if self.real in self.decoded_sequence[self.real_index + 1:]:
                self.is_trivial_prediction = True



    def insert_evaluation_data(self):
        """
            Insert the recommendation performance to database by calling the insert_evaluation method
        """
        from utils.DatabaseHelper import insert_evaluation

        insert_evaluation(user_id=self.user_id,
                          session_id=self.session_id,
                          precision=self.prm[0],
                          recall=self.prm[1],
                          mrr=self.prm[2],
                          ndcg=self.ndcg,
                          predictor_name=self.model_name,
                          trivial_prediction=self.is_trivial_prediction,
                          catalog_prediction=self.is_only_catalog_prediction,
                          catalog_count=self.catalog_counter,
                          ground_truth=self.real,
                          sequence=' '.join(map(str, self.decoded_sequence)),
                          input_sequence=' '.join(map(str, self.rnn_input)),
                          input_sequence_length=len(self.rnn_input),
                          user_sequence_length=len(self.decoded_sequence),
                          predictions=' '.join(map(str, self.recommendation)))