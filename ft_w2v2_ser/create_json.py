import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

data_path = os.path.join(str(Path(__file__).parents[1]), 'results')
result_path = os.path.join(str(Path(__file__).parents[1]), 'ft_w2v2_ser/Dataset/emodb')
DATABASE = 'emodb'
def extract_file_info(data_path, result_path, k_folds=1):
    """
    create a dataframe with file name
    """
    # load the data with extracted features from before to ensure same data split
    df = pd.read_pickle(data_path + '/extracted_features_modified_all_stats.pkl')
    df_info = df[['file', 'label']]

    # create train test split

    # split the dataset into features and target
    y = df_info['label']
    X = df_info.drop('label', axis=1)

    # perform the train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # concat X,y
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    if k_folds > 1:
        df_train = pd.concat([df_train, df_val], axis=0)
        kf = KFold(n_splits=k_folds)
        k=1
        for train_index, val_index in kf.split(df_train):
            train_fold_df = df_train.iloc[train_index]
            val_fold_df = df_train.iloc[val_index]

            # create a dictionary to save fold data split
            file_dict = {}
            train_dict = {}
            for index, row in train_fold_df.iterrows():
                train_dict[row['file']] = row['label']
                file_dict['Train'] = train_dict

            val_dict = {}
            for index, row in val_fold_df.iterrows():
                val_dict[row['file']] = row['label']
                file_dict['Val'] = val_dict

            test_dict = {}
            for index, row in df_test.iterrows():
                test_dict[row['file']] = row['label']
                file_dict['Test'] = test_dict
    
            # save file as json dictionary
            label_path = result_path + '/labels'
            if not os.path.exists(label_path):
                os.makedirs(label_path)
                
            with open(label_path + f'/{DATABASE}_info_fold_{k}.json', 'w') as fp:
                json.dump(file_dict, fp, indent=2)

            k+=1

    else:
        # create a dictionary
        file_dict = {}
        train_dict = {}
        for index, row in df_train.iterrows():
            train_dict[row['file']] = row['label']
            file_dict['Train'] = train_dict

        val_dict = {}
        for index, row in df_val.iterrows():
            val_dict[row['file']] = row['label']
            file_dict['Val'] = val_dict

        test_dict = {}
        for index, row in df_test.iterrows():
            test_dict[row['file']] = row['label']
            file_dict['Test'] = test_dict

        # save file as json dictionary
        with open(result_path + f'/labels/{DATABASE}_info.json', 'w') as fp:
            json.dump(file_dict, fp, indent=2)

    return file_dict


file = extract_file_info(data_path, result_path, k_folds=5)