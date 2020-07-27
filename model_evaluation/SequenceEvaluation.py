from model_evaluation.parent_model_evaluator import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SequenceEvaluation.py')

class Evaluator(BaseEvaluator):
    def __init__(self, user_id, model_name, predictor, item_dictionary, session_id, encoded_sequence_to_evaluate, decoded_sequence, max_sequence_length, vector_size, top_k,
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

    def sequential_evaluation(self):
        """
            Method will prepare the necessary structure for the evaluation, create next-item recommendation for each visit in the sequence, evaluate it
            and insert the results to database.

            1. Create a 3D array to keep rnn input (stack)
            2. Iterate the user sequence up to the last item. (last item is not included)
               2.1 Insert the visited-items in the user sequence to input stack, get recommendations and evaluate them.
               2.2 If we have a catalog page, add the catalog data to stack and iterate until the next dedicated item visit
                   to make predictions.
               2.3 Beware that, this iteration will not produce any recommendation if the user sequence only contains catalog
                   page visits. That's why once we reached the end of the sequence we will check the if the len(seq) - self.catalog_counter == 0,
                   then we will assign self.is_only_catalog_prediction = True to indicate the sequence only contains catalog visits, we will
                   create recommendations and evaluate them.
               Note that RNN stack should not exceed the self.max_sequence_length. If we have a sequence longer than
               self.max_sequence_length, then get the last -self.max_sequence_length items to RNN stack.

            This method is being used to evaluate the sequences WITH catalog page visits.
        """
        stack = np.full((1, self.max_sequence_length, self.vector_size), fill_value=self.mask_value, dtype='float32')

        self.catalog_counter = 0
        slicer = 0
        for index in range(len(self.encoded_sequence_to_evaluate) - 1):
            self.is_trivial_prediction = False
            self.is_only_catalog_prediction = False

            self.prm = np.zeros(len(self.metrics.values()))
            self.ndcg = 0

            self.rnn_input = self.decoded_sequence[:slicer + 1]

            temp_seq = self.encoded_sequence_to_evaluate[:slicer + 1]
            if self.encoded_sequence_to_evaluate[slicer+1].max() == 1.0:

                if slicer + 1 < len(self.encoded_sequence_to_evaluate):
                    while self.encoded_sequence_to_evaluate[slicer + 1].max() != 1.0 and temp_seq.shape[0] < len(self.decoded_sequence) - 1:
                        slicer += 1
                        self.increment_by += 1
                        self.catalog_counter += 1
                        temp_seq = self.encoded_sequence_to_evaluate[:slicer + 1]

                if slicer >= self.max_sequence_length:
                    temp_seq = temp_seq[-self.max_sequence_length:]
                    self.rnn_input = self.decoded_sequence[slicer - self.max_sequence_length + 1 : slicer + 1]

                self.seq_len = temp_seq.shape[0]
                stack[0, -self.seq_len:, :] = temp_seq[:]

                # Check if rnn stack contains only catalog pages. If yes, mark the recommendation as only_catalog_prediction
                unique_rnn_input = list(set(self.rnn_input))
                if len(unique_rnn_input) == 1:
                    if unique_rnn_input[0] == 0:
                        self.is_only_catalog_prediction = True

                self.evaluate_sequence(sequence=stack, gt=self.decoded_sequence)
                self.insert_evaluation_data()

                if self.is_trivial_prediction:
                    return

                self.increment_by += 1
            else:
                self.catalog_counter += 1
                self.increment_by += 1

            slicer += 1

            if self.increment_by == len(self.decoded_sequence):
                break

        # If we iterate all the catalog pages and have not made any predictions that it means the sequence contains only catalogs.
        # Then we will make a recommendation at the end.
        if (len(self.decoded_sequence)) - 1 - self.catalog_counter == 0:
            self.increment_by -= 1

            if slicer >= self.max_sequence_length:
                temp_seq = temp_seq[-self.max_sequence_length:]
                self.rnn_input = self.decoded_sequence[slicer - self.max_sequence_length + 1: slicer + 1]

            self.seq_len = temp_seq.shape[0]
            stack[0, -self.seq_len:, :] = temp_seq[:]
            self.is_only_catalog_prediction = True

            self.evaluate_sequence(sequence=stack, gt=self.decoded_sequence)
            self.insert_evaluation_data()
