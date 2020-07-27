import pandas as pd
import functools
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    Map the ndc_code with the generic drug name in the dataframe df
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_merged = df.merge(ndc_df[['NDC_Code', 'Non-proprietary Name']], how='left',
                         left_on='ndc_code', right_on='NDC_Code')
    df_merged = df_merged.rename(columns={'Non-proprietary Name': 'generic_drug_name'})
    df_merged = df_merged.drop(['NDC_Code', 'ndc_code'], axis=1)
    return df_merged


#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_ids = df.groupby('patient_nbr')['encounter_id'].min()
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_ids)]
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    nr_patients = df.shape[0]
    train = df.sample(n=int(0.6 * nr_patients))
    validation = df[~df[patient_key].isin(train[patient_key])].sample(n=int(0.2 * nr_patients))
    test = df[~df[patient_key].isin(pd.concat([train[patient_key], validation[patient_key]]))]
    return train, validation, test


#Question 7
def create_tf_categorical_feature_cols(categorical_col_list, vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            c, vocab_file_path, num_oov_buckets=1
        )
        tf_categorical_feature_column = tf.feature_column.indicator_column(
            tf_categorical_feature_column
        )
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col,
        default_value=default_value,
        normalizer_fn=normalizer,
        dtype=tf.float64
    )
    #tf_numeric_feature = tf.feature_column.numeric_column(
    #    key=col,
    #    default_value=default_value,
    #    normalizer_fn=lambda x: (x-MEAN)/STD,
    #    dtype=tf.float64
    #)
    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s


# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = (df[col]>=5).astype(int)
    return student_binary_prediction
