import datetime
import os

import gensim
from gensim.models import Doc2Vec
import logging
import pickle
import pandas as pd
from pandas import np
import re
import joblib
from gensim.models.doc2vec import TaggedDocument
from math import log
from utils.performance_metrics.Metrics import recall, precision, mrr
from utils.DatabaseHelper import insert_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Doc2Vec.py')

def log_event(message):
    current_time = datetime.datetime.now()
    logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

class BasicDoc2Vec(object):

    def log_event(self, message):
        current_time = datetime.datetime.now()
        logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

    def __init__(self):
        self.path_content_dataframe = f'../source_data/auxiliary_dataframe_raw.pickle'
        self.content_dataframe = None
        self.model = None
        self.characters_to_remove = '\ |\[|\]|\!|\/|\;|\:'
        self.tagged_docs = []
        self.model = None
        self.doc2vec_model_path = None

    def start(self):

        if not os.path.exists('doc2vec.model'):
            self.load_metadata()
            self.prepare_dataframe(drop_duplicate_tours=True)
            self.train_doc2vec()
        else:
            self.content_dataframe = joblib.load('doc2vec.model')


    def renew_title(self, row):
        """
            Method gets a dataframe row, process the title of the tour object by "item_id;valid_from" structure
            then returns it.
        @param row: Rows of a dataframe
        @return: New object Title
        """
        new_title = str(row.item_id) + ';' + str(row.valid_from)
        return re.sub('-', '', new_title)

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

        words = words.replace('[','')
        words = words.replace(']', ',')
        processed_words = words.split(',')[:-1]
        return processed_words

    def load_metadata(self):
        self.content_dataframe = pd.read_pickle(self.path_content_dataframe)
        self.log_event('|-- Dataframe load from the disk..')

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

        joblib.dump(self.content_dataframe, 'doc2vec_train.pickle')

        self.log_event('|----> Pre-Processing Finished..')

    def train_doc2vec(self):
        """
            Method will train a Doc2Vec Model to use for evaluation
            1. Get the item vectors from the dataframe.
            2. Those vectors will represent the "sentences of a paragraph"
            3. Iterate through vectors and tag them bu using the "tour_id;tag_index" convention so that each item
               can be found in the decoded sequence.
            4. Create a Doc2Vec object, build its vocabulart with the tagged sequences.
            5. Train the model and save the model and the tagged docs file to file system
        """
        log_event('|----> Training Doc2Vec Algorithm..')

        sequences = self.content_dataframe['item_vector'].values
        tour_ids = list(self.content_dataframe.index.values)

        for i, item in enumerate(sequences):
            self.tagged_docs.append(gensim.models.doc2vec.TaggedDocument(item, [str(tour_ids[i]) + ';' + str(i)]))

        self.model = Doc2Vec(vector_size=64, min_count=0, epochs=16, workers=16, window=1, seed=42)

        self.model.build_vocab(self.tagged_docs)

        self.model.train(self.tagged_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        self.model.save("doc2vec.model")

        joblib.dump(self.tagged_docs, 'tagged_docs.doc2vec')

        log_event('|----> Item2Vec Training is done. Model is saved to the file..')

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
    doc2vec_model = Doc2Vec.load(f'doc2vec.model')

    if not os.path.exists('evaluation_data/evaluation_sequences.data'):
        evaluation_dataframe = joblib.load(f'../source_data/common_data/evaluation_dataframe.pickle')
        evaluation_dataframe.drop(['user_log_list'], axis=1, inplace=True)
        evaluation_dataframe = remove_catalogs(dataframe=evaluation_dataframe)

        evaluation_dataframe['decoded_item_sequence'] = evaluation_dataframe['item_sequence'].copy(deep=True)
        evaluation_dataframe.drop(['catalog_item_list', 'session_start_time', 'good_catalog_items', 'sequence_length'], axis=1, inplace=True)

        sequences = evaluation_dataframe.values.tolist()
        sequences = np.array(sequences)
        joblib.dump(sequences, 'evaluation_data/evaluation_sequences.data')
    else:
        sequences = joblib.load('evaluation_data/evaluation_sequences.data')

    evaluation_creator(model=doc2vec_model,
                       model_name='base_doc2vec_model',
                       sequence_to_evaluate=sequences)

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

def evaluation_creator(model, model_name, sequence_to_evaluate):
    """
        Create an Evaluator object, prepare the constructor and start the evaluation
    @param model: Doc2Vec Model File to make predictions
    @param model_name: The name of the model (str)
    @param sequence_to_evaluate: The list of the sequences to evaluate
    """
    items_file = open(f'../source_data/item.dictionary', 'rb')
    item_dictionary = pickle.load(items_file)

    tagged_docs = joblib.load('tagged_docs.doc2vec')

    for user_id, session_id, user_sequence, decoded_individual_sequence in sequence_to_evaluate:
        evaluator = Evaluator(user_id=user_id,
                              model_name=model_name,
                              predictor=model,
                              item_dictionary=item_dictionary,
                              tagged_docs=tagged_docs,
                              session_id=session_id,
                              encoded_sequence_to_evaluate=np.array(user_sequence),
                              decoded_sequence=decoded_individual_sequence,
                              top_k=10)

        evaluator.sequential_evaluation()

class Evaluator(object):
    def __init__(self, user_id, model_name, predictor, item_dictionary, tagged_docs, session_id, encoded_sequence_to_evaluate, decoded_sequence, top_k):
        self.user_id = user_id
        self.model_name = model_name
        self.predictor = predictor
        self.item_dictionary = item_dictionary
        self.tagged_docs = tagged_docs
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
            Method receives the current user profile (clicked item(s)) and creates next-item recommendations.
            Generated recommendations are compared with the ground truth and its performance is evaluated.
        @param sequence: Visited object ID or its encoding (based on a used model)
        @param gt: The rest of the sequence after the clicked_item objectID.
        """
        user_profile = sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.doc2vec_calculate_similarity(item_id=user_profile)[:self.top_k]

        self.real = ground_truth[0]
        self.real_index = gt.index(self.real)

        # Calculate Precision, Recall and MMR
        for i, metric_function in enumerate(self.metrics.values()):
            self.prm[i] += metric_function([self.real], self.recommendation)
        # Calculate nDCG
        if 0.0 not in self.prm:
            # Then calculate nDCG
            self.calculate_ndcg()

    def doc2vec_calculate_similarity(self, item_id):
        """
            Method first iterates the tagged documents and finds the item_id given to the method. Then extracts the raw id of the item and makes predictions.
        @param item_id: The ID of the item which we want to make predictions for.
        """
        doc_tag = 0
        for tag in self.tagged_docs:
            if item_id == int(tag.tags[0].split(';')[0]):
                doc_tag = int(tag.tags[0].split(';')[-1])
                break
        try:
            similar_items = self.predictor.docvecs.most_similar(doc_tag, topn=20)
        except Exception as err:
            logging.error('Doc2Vec Algorithm could not find the given item in the vocabulary', '\n', err)
            doc_tag = -1
            print(doc_tag, '  >><<  ', item_id)
            return [-1]

        recommended_items = []
        for sim_item in similar_items[1:]:
            similar_item = int(sim_item[0].split(';')[0])
            try:
                encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(int(similar_item))]
                if similar_item not in recommended_items:
                    recommended_items.append(similar_item)
            except:
                pass
        return recommended_items[:10]

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



doc2vec = BasicDoc2Vec()
doc2vec.start()
evaluate_model()