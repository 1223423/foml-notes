Simple scikit-learn sentiment classification pipeline for fundamentals of machine learning course

# Usage

```
python train.py --csv dataset_subset_transformed.csv --predict predictors.csv --features text_object weekday_category containsMention_category --algorithm svm --ngrams 4 --cv --cm
```
## Flags

 ```--csv``` [filename.csv] _training dataset_
 
 ```--features ``` [feature1 feature2] _list of features to use in training. Notice that feature name is followed by an underscore and pandas datatype in both training set and prediction set. This is necessary for succesful encoding._
 
 ```-- algorithm ``` [svm / nb] _training algorithm_
 
 ```-- ngrams ``` [n] _Ngram range from 1 to n (not n-grams alone!), currently only words_
 
### Optional flags
 ```-- predict``` [filename.csv] _unlabeled data; must contain same feature names as in train set_
 
 ```-- crossvalidate / cv``` _cross-validation of entire training set (10 splits)_
 
 ```-- confusionmatrix / cm``` _draws basic confusion matrix (requires matplotlib)_
 
 ```-- showngrams ``` _Shows the sentence 'an apple a day keeps the doctor away' converted to the range of ngrams_

By default the text data is converted to tf-idf representation, categorical features are one hot encoded, and numerical features left as is.

# Data & wrangling utility

We're using the the Sentiment140 corpus, which can be found
* Here: http://help.sentiment140.com/for-students
* And here: https://www.kaggle.com/kazanova/sentiment140

The data contains 1.6 million tweets labelled with positive or negative sentiment.

The utility cleans the data (removes mentions, urls, html tags, symbols etc.) and creates features specifically for our purpose. It also creates a randomly sampled subset that is equally balanced across days of the week and labels for each day. We only use 10k per label per weekday, resulting in only 140k observations used. A few short tweets get destroyed during the wrangling and are removed from the set. See more in the rmd file.

## Todo
- Implement char ngrams
- Try / catch everything and general non-hardcoded safety checks
- Add other algorithms & more dynamic pipeline
- Pickle trained models for reuse
- Framework for generating automatic reports: training, evaluating & predicting across all ngram, feature, chosen split, algorithm combinations
- Document wrangling util more clearly
  - Solve that weird bug instead of having a million mutate pipes
- Reduce assumptions / parameters e.g. max features or test / train split
- Reduce dependencies where possible
- Support multiple text / object dtype features (now only uses the first available)
- More educational prints of encoding, train & evaluation
