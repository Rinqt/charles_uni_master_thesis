from model_evaluation.parent_model_evaluator import *
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SequenceEvaluationTwoBranch.py')

class Evaluator(BaseEvaluator):

    def __init__(self, user_id, model_name, predictor, item_dictionary, session_id, encoded_sequence_to_evaluate, decoded_sequence, auxiliary_sequence, max_sequence_length, vector_size, top_k,
                 mask_value):

        super().__init__(user_id=user_id,
                         model_name=model_name,
                         predictor=predictor,
                         item_dictionary=item_dictionary,
                         session_id=session_id,
                         encoded_sequence_to_evaluate=encoded_sequence_to_evaluate,
                         decoded_sequence=decoded_sequence,
                         max_sequence_length=max_sequence_length,
                         vector_size=vector_size,
                         top_k=top_k,
                         mask_value=mask_value)

        self.auxiliary_item_sequence = auxiliary_sequence

    def sequential_evaluation(self):
        """
            Method will prepare the necessary structure for the evaluation, create next-item recommendation for each visit in the sequence, evaluate it
            and insert the results to database.

            1. Create 2 3D arrays to keep rnn input (one stack for user sequence and one for content metadata)
            2. Iterate the user sequence up to the last item. (last item is not included)
               2.1 Insert the visited-items in the user sequence to first stack, apply the same for the second stack get recommendations and evaluate them.
               2.2 If we have a catalog page, add the catalog data to stack and iterate until the next dedicated item visit
                   to make predictions.
               2.3 Beware that, this iteration will not produce any recommendation if the user sequence only contains catalog
                   page visits. That's why once we reached the end of the sequence we will check the if the len(seq) - self.catalog_counter == 0,
                   then we will assign self.is_only_catalog_prediction = True to indicate the sequence only contains catalog visits, we will
                   create recommendations and evaluate them.
               Note that RNN stack should not exceed the self.max_sequence_length. If we have a sequence longer than
               self.max_sequence_length, then get the last -self.max_sequence_length items to RNN stack.

            This method is being used to evaluate the sequences WITH catalog page visits and only applicable for Model since due to its architectural difference.
        """
        stack_input_one = np.full((1, self.max_sequence_length, len(self.item_dictionary)), fill_value=-self.mask_value, dtype='float32')
        stack_input_two = np.full((1, self.max_sequence_length, self.vector_size), fill_value=self.mask_value, dtype='float32')

        self.catalog_counter = 0
        slicer = 0
        for index in range(len(self.encoded_sequence_to_evaluate) - 1):
            self.is_trivial_prediction = False
            self.is_only_catalog_prediction = False

            self.prm = np.zeros(len(self.metrics.values()))
            self.ndcg = 0

            input_one_seq = self.encoded_sequence_to_evaluate[:slicer + 1]
            input_two_seq = self.auxiliary_item_sequence[:slicer + 1]

            self.rnn_input = self.decoded_sequence[:slicer + 1]

            if self.encoded_sequence_to_evaluate[slicer].max() == 1.0:

                if slicer + 1 < len(self.encoded_sequence_to_evaluate):
                    while self.encoded_sequence_to_evaluate[slicer + 1].max() != 1.0 and input_one_seq.shape[0] < len(self.decoded_sequence) - 1:
                        slicer += 1
                        self.increment_by += 1
                        self.catalog_counter += 1
                        input_one_seq = self.encoded_sequence_to_evaluate[:slicer + 1]
                        input_two_seq = self.auxiliary_item_sequence[:slicer + 1]

                if slicer >= self.max_sequence_length:
                    input_one_seq = input_one_seq[-self.max_sequence_length:]
                    input_two_seq = input_two_seq[-self.max_sequence_length:]
                    self.rnn_input = self.decoded_sequence[slicer - self.max_sequence_length + 1 : slicer + 1]

                self.seq_len = input_one_seq.shape[0]

                stack_input_one[0, -self.seq_len:, :] = input_one_seq[:]
                stack_input_two[0, -self.seq_len:, :] = input_two_seq[:]

                self.evaluate_two_branch_sequence(stack_input_one, stack_input_two, self.decoded_sequence)
                self.insert_evaluation_data()

                if self.is_trivial_prediction:
                    return self.prm, self.ndcg

                self.increment_by += 1
            else:
                self.catalog_counter += 1
                self.increment_by += 1

            slicer += 1

            if self.increment_by == len(self.decoded_sequence):
                break

        if (len(self.decoded_sequence)) - 1 - self.catalog_counter == 0:
            # catalog prediction
            self.increment_by -= 1

            if slicer >= self.max_sequence_length:
                input_one_seq = input_one_seq[-self.max_sequence_length:]
                input_two_seq = input_two_seq[-self.max_sequence_length:]
                self.rnn_input = self.decoded_sequence[slicer - self.max_sequence_length + 1: slicer + 1]

            self.seq_len = input_one_seq.shape[0]

            stack_input_one[0, -self.seq_len:, :] = input_one_seq[:]
            stack_input_two[0, -self.seq_len:, :] = input_two_seq[:]
            self.is_only_catalog_prediction = True

            self.evaluate_two_branch_sequence(stack_input_one, stack_input_two, self.decoded_sequence)
            self.insert_evaluation_data()


    def evaluate_two_branch_sequence(self, sequence, auxiliary_sequence, gt):
        """
            Method wil assign the user_profile and ground_truth based on the user sequence.
            User profile will be sent to the RNN for the predictions
            Then method will calculate precision, recall and MRR first. If prediction has performance score
            bigger than zero, we will calculate the nDCG Score.
        :param sequence: User sequence to evaluate
        :param auxiliary_sequence: Tour data for the objects inside the user sequence
        :param gt: Ground Truth
        """
        input_one_user_profile = sequence
        input_two_user_profile = auxiliary_sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.lstm_calculate_two_branch_similarity(user_sequence=input_one_user_profile, auxiliary_data=input_two_user_profile)[:self.top_k]

        self.real = ground_truth[0]
        self.real_index = gt.index(self.real)

        # Calculate Precision, Recall and MMR
        for i, metric_function in enumerate(self.metrics.values()):
            self.prm[i] += metric_function([self.real], self.recommendation)

        if 0.0 not in self.prm:
            # Then calculate nDCG
            self.calculate_ndcg()


    def lstm_calculate_two_branch_similarity(self, user_sequence, auxiliary_data):
        """
            Get the Top-K Recommendation by using the user profile.
            Note that recommended items will be item-dictionary encoding. So they need to be
            transformed to their original objectIDs.

            This is an override method since the RNN uses two input. ([user_sequence, auxiliary_data])
        :param user_sequence: User profile which contains the encoded implicit feedback
        :param auxiliary_data: Tour details which contains the encoded tour object information
        :return: Top-K Recommendation List
        """
        try:
            similar_items = self.predictor.predict([user_sequence, auxiliary_data])
        except Exception as err:
            logging.error('LSTM Algorithm could not find the given item in the vocabulary', '\n', err)
            return [-1]

        top_k = (-similar_items).argsort()[0][:self.top_k]

        recommended_items_set = []
        for rec_item in top_k:
            if rec_item in self.item_dictionary:
                encoded_item = self.item_dictionary.get(rec_item)
                recommended_items_set.append(encoded_item)

        return recommended_items_set[:self.top_k]