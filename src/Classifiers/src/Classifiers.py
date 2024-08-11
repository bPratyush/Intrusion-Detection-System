import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

one_hot_encoder = None

def load_file(path, mode, is_attack=1, label=1, folder_name='Bi/', sliceno=0, verbose=True):
    global one_hot_encoder
    
    columns_to_drop_packet = ['timestamp', 'src_ip', 'dst_ip']
    
    dataset = pd.read_csv(path, low_memory=False)
    dataset = dataset.loc[dataset['is_attack'] == is_attack]
    
    dataset.drop(columns=columns_to_drop_packet, inplace=True)
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='mqtt')))]
    dataset = dataset.fillna(-1)
    
    # Handle non-numeric values
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])  # Fill NaN values with mode
    
    x = dataset.values.astype(float)  # Convert values to float
    
    with open(folder_name + 'instances_count.csv', 'a') as f:
        f.write('all, {}, {} \n'.format(path, x.shape[0]))
    
    x = np.unique(x, axis=0)
    
    with open(folder_name + 'instances_count.csv', 'a') as f:
        f.write('unique, {}, {} \n'.format(path, x.shape[0]))
    
    if (mode == 1 and x.shape[0] > 100000) or (mode == 2 and x.shape[0] > 50000):
        temp = x.shape[0] // 10
        start = sliceno * temp
        end = start + temp - 1 
        x = x[start:end,:] 
        with open(folder_name + 'instances_count.csv', 'a') as f:
            f.write('Start, {}, End, {} \n'.format(start, end))
    elif mode == 0:
        if x.shape[0] > 15000000:
            temp = x.shape[0] // 400
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(folder_name + 'instances_count.csv', 'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
        elif x.shape[0] > 10000000:
            temp = x.shape[0] // 200
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(folder_name + 'instances_count.csv', 'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
        elif x.shape[0] > 100000:
            temp = x.shape[0] // 10
            start = sliceno * temp
            end = start + temp - 1 
            x = x[start:end,:] 
            with open(folder_name + 'instances_count.csv', 'a') as f:
                f.write('Start, {}, End, {} \n'.format(start, end))
            
    y = np.full(x.shape[0], label)
    
    with open(folder_name + 'instances_count.csv', 'a') as f:
        f.write('slice, {}, {} \n'.format(path, x.shape[0]))
        
    return x, y

def classify_sub(classifier, x_train, y_train, x_test, y_test, cm_file_name, summary_file_name, classifier_name, verbose=True):
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    
    cm = pd.crosstab(y_test, pred)
    cm.to_csv(cm_file_name)    
    
    pd.DataFrame(classification_report(y_test, pred, output_dict=True)).transpose().to_csv(summary_file_name)
    
    if verbose:
        print(classifier_name + ' Done.\n')
    
    del classifier
    del pred
    del cm
    
def classify(random_state, x_train, y_train, x_test, y_test, folder_name, prefix="", verbose=True):
    confusion_matrix_folder = os.path.join(folder_name, 'Confusion_Matrix/') 
    summary_folder =  os.path.join(folder_name, 'Summary/') 

    if os.path.isdir(confusion_matrix_folder) == False:
            os.mkdir(confusion_matrix_folder)
    if os.path.isdir(summary_folder) == False:
            os.mkdir(summary_folder)
            
    # 1- Linear
    linear_classifier = LogisticRegression(random_state=random_state)
    classify_sub(linear_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_linear.csv', 
                 summary_folder + prefix + '_summary_linear.csv',
                 'Linear',
                 verbose)
       
    # 2- KNN
    knn_classifier = KNeighborsClassifier()
    classify_sub(knn_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_knn.csv', 
                 summary_folder + prefix + '_summary_knn.csv',
                 'KNN',
                 verbose)
    
    #3- RBF SVM
    kernel_svm_classifier = SVC(kernel='rbf', random_state=random_state, gamma='scale')
    classify_sub(kernel_svm_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_kernel_svm.csv', 
                 summary_folder + prefix + '_summary_kernel_svm.csv',
                 'SVM',
                 verbose)
    
    #4- Naive Bayes
    naive_classifier = GaussianNB()
    classify_sub(naive_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_naive.csv', 
                 summary_folder + prefix + '_summary_naive.csv',
                 'Naive',
                 verbose)

    #5- Decision Tree
    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
    classify_sub(decision_tree_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_decision_tree.csv', 
                 summary_folder + prefix + '_summary_decision_tree.csv',
                 'Decision Tree',
                 verbose)
    
    #6- Random Forest
    random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=random_state)
    classify_sub(random_forest_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_random_forest.csv', 
                 summary_folder + prefix + '_summary_random_forest.csv',
                 'Random Forest',
                 verbose)

    # 7- Linear SVM 
    svm_classifier = LinearSVC(random_state=random_state)
    classify_sub(svm_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_svm.csv', 
                 summary_folder + prefix + '_summary_svm.csv',
                 'SVM',
                 verbose)

    # 8- SVM with RBF kernel
    svm_rbf_classifier = SVC(kernel='rbf', random_state=random_state, gamma='auto')
    classify_sub(svm_rbf_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_svm_rbf.csv', 
                 summary_folder + prefix + '_summary_svm_rbf.csv',
                 'SVM with RBF kernel',
                 verbose)

    # 9- XGBoost
    xgb_classifier = XGBClassifier(random_state=random_state)
    classify_sub(xgb_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_xgb.csv', 
                 summary_folder + prefix + '_summary_xgb.csv',
                 'XGBoost',
                 verbose)

    # 10- CatBoost
    catboost_classifier = CatBoostClassifier(random_state=random_state, verbose=False)
    classify_sub(catboost_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_catboost.csv', 
                 summary_folder + prefix + '_summary_catboost.csv',
                 'CatBoost',
                 verbose)

    # 11- LightGBM
    lgbm_classifier = LGBMClassifier(random_state=random_state)
    classify_sub(lgbm_classifier, 
                 x_train, y_train, 
                 x_test, y_test, 
                 confusion_matrix_folder + prefix + '_cm_lgbm.csv', 
                 summary_folder + prefix + '_summary_lgbm.csv',
                 'LightGBM',
                 verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--output', default='Classification_Bi')
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()
    
    for slice_number in range(10):
        prefix = ''
        if args.mode == 1:
            prefix = 'uniflow_' 
        elif args.mode == 2:
            prefix = 'biflow_'
        
        if args.verbose:
            print('Starting Slice #: {}'.format(slice_number))
            print('Start Classification')
            
        random_state = 0
        folder_name = '{}_{}/'.format(args.output, slice_number)
        
        if os.path.isdir(folder_name) == False:
            os.mkdir(folder_name)
            
        x, y = load_file('/Users/bpratyush/Downloads/packet_features/normal.csv', 
                         args.mode, 
                         0, 0, 
                         folder_name, 
                         slice_number,
                         args.verbose)
        
        x_temp, y_temp = load_file('/Users/bpratyush/Downloads/packet_features/scan_A.csv',
                                   args.mode, 
                                   1, 1, 
                                   folder_name,
                                   slice_number,
                                   args.verbose)
        
        x = np.concatenate((x, x_temp), axis=0)
        y = np.append(y, y_temp)
        del x_temp, y_temp
        
        x_temp, y_temp = load_file('/Users/bpratyush/Downloads/packet_features/scan_sU.csv', 
                                   args.mode, 
                                   1, 2, 
                                   folder_name,
                                   slice_number,
                                   args.verbose)
        
        x = np.concatenate((x, x_temp), axis=0)
        y = np.append(y, y_temp)
        del x_temp, y_temp
                
        x_temp, y_temp = load_file('/Users/bpratyush/Downloads/packet_features/sparta.csv', 
                                   args.mode, 
                                   1, 3,
                                   folder_name,
                                   slice_number,
                                   args.verbose)
        
        x = np.concatenate((x, x_temp), axis=0)
        y = np.append(y, y_temp)
        del x_temp, y_temp
                
        x_temp, y_temp = load_file('/Users/bpratyush/Downloads/packet_features/mqtt_bruteforce.csv', 
                                   args.mode,
                                   1, 4, 
                                   folder_name,
                                   slice_number,
                                   args.verbose)
        
        x = np.concatenate((x, x_temp), axis=0)
        y = np.append(y, y_temp)
        del x_temp, y_temp
                
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            test_size=0.25,
                                                            random_state=42)
        
        classify(random_state, x_train, y_train, x_test, y_test, 
                 folder_name, "slice_{}_no_cross_validation".format(slice_number), args.verbose)
       
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        
        counter = 0
        for train, test in kfold.split(x, y):
            classify(random_state, x[train], y[train], x[test], y[test], 
                     folder_name, "slice_{}_k_{}".format(slice_number, counter), args.verbose)
            counter += 1
