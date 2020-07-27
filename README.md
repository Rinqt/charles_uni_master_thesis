Repository for Charles University Software and Data Engineering Master's Degree Thesis.

In this thesis, we focus on using Recurrent Neural Network with Long Short Term Memory and try to improve the performance of session-aware recommendations using real-life user implicit feedback and content metadata.  One of our work’s contributions is to help increase the recommendation quality on small e-commerce domains where the density of anonymous users is high, and the length of a user session is short.

Different types of RNN models that use various encoding strategies have been implemented and evaluated on a real-life dataset.  It has been shown that usage of implicit feedback affect recommendation quality directly.  Throughout the thesis, four evaluation metric is used and between all the metrics nDCG is given the most priority due to its success in evaluating the position of ranked recommendations. We addressed the problem of data sparsity in small e-commerce domain.  Our experiments show that utilizing specific implicit feedback increases the quality of the recommendations.  It is possible that our experiment can be applied for all the other data sets that satisfy certain conditions like logging user implicit feedback. 

# Project Structure

```bash
Parent Folder
│   environment.yml
│   main.py
│
├───baseline_models
│   │   CosineSimilarity.py
│   │   Doc2Vec.py
│   │   Item2Vec.py
│   │
│   └───evaluation_data
│           evaluation_sequences.data
│
├───database_scripts
|	    table_content_base_tour_details.sql
│       table_implicit_user_feedback.sql
│       table_lstm_models.sql
│       table_model_evaluation.sql
│       table_model_performance.sql
|
├───docs
├───lstm_models
│       model_01.py
│       model_02.py
│       model_03.py
│       model_04.py
│       model_05.py
│       model_06.py
│       model_07.py
│       parent_lstm_model.py
│
├───model_data
├───model_evaluation
│       BasicSequenceEvaluation.py
│       parent_model_evaluator.py
│       SequenceEvaluation.py
│       SequenceEvaluationItem2Vec.py
│       SequenceEvaluationTwoBranch.py
│
├───source_data
│   └───common_data
│       └───folds
└───utils
    │   AuxiliaryDataProcessor.py
    │   DatabaseHelper.py
    │   item2vec.model
    │   TrainingDataProcessor.py
    │
    └───performance_metrics
        │   Metrics.py
```

Project contains several folder to keep necessary files. 

1. **baseline_models:** Contains three baseline model that we compare our solutions with.
2. **database_scripts:** Contains SQL scripts to create necessary tables.
3. **docs:** Contains the PDF version of the thesis.
4. **lstm_models:** Contains implemented RNN LSTM Models along with their parent class.
5. **model_data:** Folder to contain training data for the implemented models. At the beginning this folder is empty. Once the code runs, it will create the necessary files/folder.
6. **model_evaluation:** Contains evaluation scripts along with their parent class. Due to different type of models, there are couple evaluation scripts.
7. **source_data:** Initially empty. Once the TraningDataProcessor.py and AuxiliaryDataProcessor.py runs, they will create the necessary files that can be use to train all RNN models. Additionally, there is a sub folder called **common_data** which must be created before running the code. This folder contains another sub folder named **folds**. TraningDataProcessor.py will create K-Fold split and place the input data under folds folder.
8. **utils:** Contains utility scripts and files along with performance metrics

# Configuration

There are certain steps to take before running the code in your machine. Note that repository source code, does not contain all the necessary data. Once you clone the project, make sure that all the folders mentioned in Project Structure is created.

The dataset we used is a real-life data. Due to confidentiality, we uploaded small portion of the source data and changed the user_ids to random numbers.

The code is developed on a Windows machine. Thus, we do not guarantee its performance or successful run under other operation systems.

Once the project is cloned, follow the below steps to configure your machine to run the code.

You can also watch the [tutorial]: https://youtu.be/kg9RzUl1Eug for the required steps. 

## Prerequisites

- Python 3.7
- Nvidia CUDA Drivers
- Anaconda and Pip
- Graphviz for saving architecture of the machine learning model

## Importing the Database

1. Locate the SQL scripts under the **database_scripts** folder and import it to an SQL Database. We will illustrate the database import on MySQL Workbench, since it was used during the development.

   - Open MySQL Workbench and navigate File > Run SQL Script. Select the script file and open it.

   - Choose a database to import the table and click run. 

   - Reply the same step for each script.

     Note that the database name we used for the code is "travel_agency". If you will be using a different database name, update the DatabaseHelper.py accordingly.
     
     

2. Locate the DatabaseHelper.py located under utils folder. Open the script and replace the database config dictionary with your environment values.

```python
   config = {
       'user': 'insert_user_name_here',
       'password': 'insert_password_here',
       'host': '127.0.0.1',
       'database': 'travel_agency',
       'raise_on_warnings': True,
       'auth_plugin': 'mysql_native_password'
   }
```

   

## Importing the Virtual Environment

1. Import the virtual environment by using the environment.yml file located under the project folder.

2. Open Anaconda prompt and type the following to import the environment.

   ```cmd
   conda env create -f rec_sys_thesis.yml
   ```

   Note that importing the environment might take time based on the network speed. 

3. Once the import is complete, activate the virtual environment.

4. The RNN Models are trained by Tensorflow 2.2 on CUDA. Make sure your CUDA drivers are installed (https://www.tensorflow.org/install/gpu).

## Running the Code

Make sure that virtual environment is activated and bind to the project.

The code is developed by using PyCharm with Anaconda. The configuration tutorial will cover the steps to run the code and show the PyCharm configuration

1. In order to train and evaluate the models, first you need to create necessary data.

2. First run TraningDataProcessor.py script. The script will create required data to train and evaluate RNN.

   ​	Excepted outcome:

   ![00](/images/00.png)

3. Then, run AuxiliaryDataProcessor.py file to create Auxiliary Input data for Model 6 and 7.

   ​	Expected outcome:

   ![01](/images/01.png)

4. Finally run the main.py.

   ​	Expected outcome

   ![02](/images/02.png)

   Note that each method is described with comment blocks and will used default values we set during the research. 

For more details regarding the research navigate to docs/yos_master_thesis.pdf. To get more information about the source code, check the source code.
