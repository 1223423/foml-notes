# Author: Benas Cernevicius
# Last updated: 31/10/2020

import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def extract_features(df, features):
    # Returns df with desired features and converts them to appropriate datatype
    # Could go wrong: not checking for non-existing features
    ef = df[features].copy()
    for feature in features:
        dtype = feature.split(sep='_')
        ef[feature] = ef[feature].astype(dtype[-1:][0])
    return ef

def construct_feature_dtypes(features):
    dtypes = {
        'category' : [],
        'object' : [],
        'int64' : [],
        'float64' : [],
        'bool' : []
    }
    for feature in features:
        dtype = feature.split(sep='_')
        dtypes[dtype[-1:][0]].append(feature)
    return dtypes

def select_classifier(algorithm):
    return {
        'svm': svm.LinearSVC(),
        'nb': MultinomialNB(),
        'logreg' : LogisticRegression(random_state=0, max_iter = 20000)
    }[algorithm]

if __name__ == "__main__":


    # Parse runtime arguments (not in use right now, will implement with features)
        parser = argparse.ArgumentParser()
        parser.add_argument('--crossvalidate','--cv', help = "perform cross-validation on entire dataset yes/no", action="store_true", required=False)
        parser.add_argument('--csv', help = "training dataset path eg: data/train.csv", default = "no", type = str, required=True)
        parser.add_argument('--predict', help = "prediction dataset path eg: data/predict.csv", default = None, type = str, required=False)
        parser.add_argument('--features', help = "train / predict features e.g. text_object gender_category age_int64", nargs = "+", default = None, type = str, required=True)
        parser.add_argument('--algorithm', help = "choose algorithms e.g. svm or nb", default = None, type = str, required = True)
        parser.add_argument('--ngrams', help = "use ngrams up to x", default = 1, type = int, required = True)
        parser.add_argument('--showngrams', help = "show example of ngram encoding", action="store_true", required = False)
        parser.add_argument('--confusionmatrix','--cm', help = "plot confusion matrix", action="store_true", required = False)
        args = parser.parse_args()

    # Parameters
        use_features = args.features
        ngram_max = args.ngrams
        max_features = 100000
        test_size = 0.2
        classifier = select_classifier(args.algorithm)

    # Load data
        # Training set
        training_data = pd.read_csv(args.csv, delimiter=',')
        feature_datatypes = construct_feature_dtypes(use_features)

        # Safety check: does data have NA values? If so, warn, remove, and continue
        text_feature = feature_datatypes['object'][0]
        if(training_data[text_feature].isnull().values.any()):
            print("[WARNING] dataset contains missing values!")
            print("[WARNING] Dropped",training_data[text_feature].isnull().sum(), "NA rows")
            training_data = training_data[training_data[text_feature].notna()]

        # Prediction set
        if(args.predict):
            pred_data = pd.read_csv(args.predict, delimiter=',')
            pred_features = extract_features(pred_data, use_features)

    # Feature extraction
        features = extract_features(training_data, use_features)
        target = training_data.label

    # Data split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state=1337)

    # Preprocessing
        # Currently only the first text feature will be used
        preprocess = ColumnTransformer(
            [
            ('text_tfidf', TfidfVectorizer(max_features = max_features, stop_words = 'english', ngram_range=(1,ngram_max)), text_feature),
            ('onehot_category', OneHotEncoder(dtype='int', handle_unknown='ignore'), feature_datatypes['category'])
            ],
            remainder='drop')

        # Pipe preprocessing -> classifier
        model = make_pipeline(
            preprocess,
            classifier)

    # Training
        model.fit(X_train, y_train)
        
        if(args.showngrams):
            print("\n\nExample of 'an apple a day keeps the doctor away' text using this ngram range:\n\n",TfidfVectorizer(max_features = MAX_FEAT_DESCP, stop_words = 'english', ngram_range=(1,ngram_max)).fit(["an apple a day keeps the doctor away"]).vocabulary_)
        
    # Evaluation
        # Baseline
        dummy_clf = DummyClassifier(strategy="uniform")
        dummy_clf.fit(X_train, y_train)
        baseline = dummy_clf.score(X_test, y_test)
        print(str('Baseline accuracy: {:04.2f}'.format(baseline*100)) + '%')

        # Accuracy
        accuracy = metrics.accuracy_score(model.predict(X_test), y_test)
        print(str('Model accuracy: {:04.2f}'.format(accuracy*100)) + '%')

        # Cross-validation (entire-set)
        if(args.crossvalidate):
            print("Cross validating (10 splits) across entire dataset... (Takes longer)")
            cross_val_score = (cross_val_score(model, features, target, cv=10, scoring = 'accuracy').mean())
            print(str('Cross-validation: {:04.2f}'.format(cross_val_score*100)) + '%')
            print("Accuracy: %0.3f (+/- %0.3f)" % (cross_val_score.mean(), cross_val_score.std() * 2))
        
        # Confusion matrix
        if(args.confusionmatrix):
            matrix = metrics.plot_confusion_matrix(model, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='all')
            plt.title('Confusion matrix for ' + args.algorithm + ' classifier')
            plt.show()
                
        # Predictions
        if(args.predict):
            print("Predictions:", model.predict(pred_features))




