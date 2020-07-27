from model_evaluation.parent_model_evaluator import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BasicSequenceEvaluation.py')

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
            Method will prepare the necessary structure for the evaluation, create next-item recommendations for each item in the sequence, evaluate it
            and insert the results to database.

            1. Create a 3D array to keep rnn input (stack)
            2. Iterate the user sequence up to the last item. (last item is not included)
               2.1 Insert the visited-items in the user sequence to input stack, get recommendations and evaluate them.
                   Note that RNN stack should not exceed the self.max_sequence_length. If we have a sequence longer than
                   self.max_sequence_length, then get the last -self.max_sequence_length items to RNN stack.

            This method can be only used to evaluate sequences which has NO catalog visit.
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

            if slicer >= self.max_sequence_length:
                temp_seq = temp_seq[-self.max_sequence_length:]
                self.rnn_input = self.decoded_sequence[slicer - self.max_sequence_length + 1: slicer + 1]

            self.seq_len = len(temp_seq)
            stack[0, -self.seq_len:, :] = np.array(temp_seq[:]).reshape(self.seq_len, 1)

            self.evaluate_sequence(stack, self.decoded_sequence)

            self.insert_evaluation_data()

            if self.is_trivial_prediction:
                return

            self.increment_by += 1
            slicer += 1

            if self.increment_by == len(self.decoded_sequence):
                break