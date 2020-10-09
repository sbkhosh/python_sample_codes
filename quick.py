#!/usr/bin/python3

import numpy as np
import multiprocessing as mp

# computing the 60 min volatility given 1min interval data
df_hv = df[[el for el in df.columns if '_ret' in el]].groupby(pd.Grouper(freq='60min',level='Dates')).apply(np.std) * np.sqrt(60)

# formatting output
map(lambda x: '{0:.2f}'.format(x),list(predictions))




###############################
# (0) Chi-squared statistics
###############################

# Chi-square statistics examines the independence of two categorical vectors. That is,
# the statistic is the difference between the observed number of observations in each
# class of a categorical feature and what we would expect if that feature was independ‐
# ent (i.e., no relationship) with the target vector:

# Chi^2 = sum_i (O_i-E_i)^2 / E_i

# where O_i is the number of observations in class i and E_i is the number of observations
# in class i we would expect if there is no relationship between the feature and target
# vector.

# A chi-squared statistic is a single number that tells you how much difference exists
# between your observed counts and the counts you would expect if there were no rela‐
# tionship at all in the population. By calculating the chi-squared statistic between a
# feature and the target vector, we obtain a measurement of the independence between
# the two. If the target is independent of the feature variable, then it is irrelevant for our
# purposes because it contains no information we can use for classification. On the
# other hand, if the two features are highly dependent, they likely are very informative
# for training our model.

###############################
# (1) swap rows and colums in matrix (without np transpose)
###############################
mat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

list(zip(*mat))

###############################
# (2) flatten any list/array
###############################
lst = [[1,2,3],[4,5,6,],[7,8,9]]
[ y for x in lst for y in x  ]

###############################
# (3) gcd
###############################
def gcd(m,n):
    while(m%n) != 0:
        old_m = m
        old_n = n

        m = old_n
        n = old_m%old_n
    return(n)

###############################
# (4) numpy stuff
###############################
a = np.arange(10)
np.where(a < 5, a, 10*a)
        

###############################
# (5) check two words are anagrams
###############################
str1,str2 = 'earth','heart'
print(sum(list(map(ord,str1))))
print(sum(list(map(ord,str2))))

###############################
# (6) protect password at screen
###############################
from getpass import getpass

psswd = getpass('give your password: ')
print(psswd)

###############################
# (7) count letters
###############################
def count_letters(word):
    letters = list(word)
    set_letters = set(letters)
    dct_letters = {}
    for el in set_letters:
        dct_letters[str(el)] = letters.count(el)
    return(dct_letters)

###############################
# (8) scatter plot
###############################
def scatter_plot(df):
    scatter_matrix(df,alpha = 0.3,figsize = (6,6),diagonal = 'kde')
    plt.show()

###############################
# (9) quick plot
###############################
def plots():
    df = pd.read_csv('dudybar.txt', sep=" ", header=None)
    df.columns = ["time", "up", "down"]
    plt.scatter(x=df["up"], y=df["down"],alpha=0.3,s=30)
    plt.xlabel('up')
    plt.ylabel('down')
    plt.title('correlation')
    plt.show()

###############################
# (10) list & dict
###############################
lst = [['ddd', 23.1], ['bbb', 33.1], ['aaa', 43.1], ['ccc', 23.1]]  
dict_lst = { k[0]: k[1] for k in lst }
print(dict_lst)

sorted_dict_lst = []
for w in sorted(dict_lst, key=dict_lst.get, reverse=True):
    sorted_dict_lst.append([w,dict_lst[w]])
print(sorted_dict_lst)    

###############################
# (11) plot/subplot
###############################
plt.figure(figsize=(10,15))
ax1 = plt.subplot(2, 1, 1)
ax1.set_title(myf_csv+'- close',fontsize=16)
df['Close'].plot()

ax2 = plt.subplot(2, 1, 2, sharex = ax1)
ax2.set_title(myf_csv+'- vol',fontsize=16)
df['STD'].plot()

###############################
# (12) timing with argument
###############################
def timeit1():
    s = time.time()
    for i in xrange(750000):
        z=i**.5
    print "Took %f seconds" % (time.time() - s)

def timeit2(arg=math.sqrt):
    s = time.time()
    for i in xrange(750000):
        z=arg(i)
    print "Took %f seconds" % (time.time() - s)

###############################
# (13) L2 norm
###############################
x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)

# (14) create dataframe
###############################
data = {'name' : ['AA', 'IBM', 'GOOG'],
        'date' : ['2001-12-01', '2012-02-10', '2010-04-09'],
        'shares' : [100, 30, 90],
        'price' : [12.3, 10.3, 32.2]}

df = pd.DataFrame(data)
df = df.set_index(['date'])
df = df.drop(['shares','price'], axis = 1)

###############################
# (15) open file and read
###############################
with open('test.txt','r') as f:
    fcontents = f.read()

words = fcontents.split(' ')
wc = len(words)
print('number of words = {}'.format(wc))

###############################
# (16) count number of cpus
###############################
num_cores = mp.cpu_count()
print(num_cores)


###############################
# (17) vectorize
###############################
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

add_100 = lambda i: i + 100

vectorized_add_100 = np.vectorize(add_100)

vectorized_add_100(matrix)

###############################
# (18) rolling statistics
###############################
rolling_mean = timeseries.rolling(window=252).mean()
rolling_std = timeseries.rolling(window=252).std()

###############################
# (19) linalg
###############################

# Return matrix rank
np.linalg.matrix_rank(matrix)

# Return determinant of matrix
np.linalg.det(matrix)

# Return diagonal elements
matrix.diagonal()

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Calculate dot product
np.dot(vector_a, vector_b)

# Calculate dot product
vector_a @ vector_b

# Add two matrices
np.add(matrix_a, matrix_b)

# Subtract two matrices
np.subtract(matrix_a, matrix_b)

# Multiply two matrices element-wise
matrix_a * matrix_b

# Calculate inverse of matrix
np.linalg.inv(matrix)

# Multiply two matrices
np.dot(matrix_a, matrix_b)

# Multiply two matrices
matrix_a @ matrix_b

###############################
# (20) random
###############################

# Set seed
np.random.seed(0)
# Generate three random floats between 0.0 and 1.0
np.random.random(3)
# Generate three random integers between 1 and 10
np.random.randint(0, 11, 3)
# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)
# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)

###############################
# (21) create simulated data
###############################

# for regression
# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100,
                                                 n_features = 3,
                                                 n_informative = 3,
                                                 n_targets = 1,
                                                 noise = 0.0,
                                                 coef = True,
                                                 random_state = 1)

# for classification
# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,
                                       n_features = 3,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.25, .75],
                                       random_state = 1)

# for clustering
# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
                              n_features = 2,
                              centers = 3,
                              cluster_std = 0.5,
                              shuffle = True,
                              random_state = 1)


###############################
# (22) query from a SQL database
###############################

import pandas as pd
from sqlalchemy import create_engine

# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

###############################
# (23) one hot encoding
###############################
pd.get_dummies(df, columns=['Sex']).head()

###############################
# (24) encoding with sklearn
###############################
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
obj_df["make_code"] = lb_make.fit_transform(obj_df["make"])
obj_df[["make", "make_code"]].head(11)

###############################
# (25) filtering dataframe
###############################
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

###############################
# (27) rename column
###############################
dataframe.rename(columns={'PClass': 'Passenger Class'})

###############################
# (28) delete column
###############################
dataframe.drop('Age', axis=1).head(2)

###############################
# (29) append row to dataframe
###############################
lst = df.iloc[-1]
df.append(lst,ignore_index=True)

###############################
# (30) group rows by the values of the column 'Sex', calculate mean of each group
###############################
df.groupby('Sex').mean()

###############################
# group rows, calculate mean
###############################
dataframe.groupby(['Sex','Survived'])['Age'].mean()

###############################
# (31) resample by week, calculate sum per week
###############################
dataframe.resample('W').sum()

###############################
# (32) create dataframe - time indexed
###############################
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
dataframe = pd.DataFrame(index=time_index)
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

###############################
# (33) apply function to dataframe
###############################
def uppercase(x):
    return(x.upper())

dataframe['Name'].apply(uppercase)

# (34) concatenate dataframes
# along row axis
pd.concat([dataframe_b,dataframe_b],axis=0,ignore_index=True)
# along column axis
pd.concat([dataframe_b,dataframe_b],axis=1,ignore_index=True)

###############################
# (35) scaling data sklearn
###############################
from sklearn.preprocessing import MinMaxScaler
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

minmax_scale = MinMaxScaler(feature_range=(0, 1)) # other scalers can be used
scaled_feature = minmax_scale.fit_transform(feature)

###############################
# (36) normalizing data
###############################
from sklearn.preprocessing import Normalizer
features_l2_norm = Normalizer(norm="l2").transform(features)

###############################
# (37) create a function to return index of outliers
###############################
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return(np.where((x > upper_bound) | (x < lower_bound)))

###############################
# (38) create feature based on boolean condition
###############################
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)    
    
###############################
# (39) bin feature
###############################
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

np.digitize(age, bins=[20,30,64])

###############################
# (40) grouping using clustering
###############################
from sklearn.cluster import KMeans
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
clusterer = KMeans(3, random_state=0)
clusterer.fit(features)
dataframe["group"] = clusterer.predict(features)

###############################
# (41) keep only observations that are not (denoted by ~) missing
###############################
features[~np.isnan(features).any(axis=1)]

###############################
# (42) handling imbalanced data
###############################
# Many algorithms in scikit-learn offer a parameter to weight classes during training to
# counteract the effect of their imbalance. While we have not covered it yet, RandomFor
# estClassifier is a popular classification algorithm and includes a class_weight
# parameter. You can pass an argument specifying the desired class weights explicitly:



# Create weights for imbalanced array where lot of 1s and few 0s
# give more weights to 0s than to 1s
weights = {0: .9, 1: 0.1}
RandomForestClassifier(class_weight=weights)

###############################
# (43) text handling with re
###############################
def replace_letters_with_X(string: str) -> str:
    return(re.sub(r"[a-zA-Z]", "X", string))

###############################
# (44) html handling with BeautifulSoup 
###############################
from bs4 import BeautifulSoup
html = """
<div class='full_name'><span style='font-weight:bold'>
Masego</span> Azra</div>"
"""
soup = BeautifulSoup(html, "lxml")
soup.find("div", { "class" : "full_name" }).text    

###############################
# (45) datetime handling
###############################
dt = '03-04-2005 11:35 PM'
pd.to_datetime(dt,format='%d-%m-%Y %I:%M %p')

# create daterange
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

dates.dt.weekday_name

###############################
# (46) timezone handling
###############################
date = pd.Timestamp('2017-05-01 06:00:00')
date_in_london = date.tz_localize('Europe/London')
date_in_tokyo = date_in_london.tz_convert('Asia/Tokyo')

# show all timezones
from pytz import all_timezones
print(all_timezones)

###############################
# (47) creating lagged feature
###############################
dataframe = pd.DataFrame()
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

#        dates  stock_price  previous_days_stock_price
# 0 2001-01-01          1.1                        NaN
# 1 2001-01-02          2.2                        1.1
# 2 2001-01-03          3.3                        2.2
# 3 2001-01-04          4.4                        3.3
# 4 2001-01-05          5.5                        4.4

###############################
# (48) PCA
###############################
# Principal component analysis (PCA) is a popular linear dimensionality reduction
# technique. PCA projects observations onto the (hopefully fewer) principal compo‐
# nents of the feature matrix that retain the most variance. PCA is an unsupervised
# technique, meaning that it does not use the information from the target vector and
# instead only considers the feature matrix.

# PCA is able to reduce the dimensionality of our feature matrix (e.g., the number of
# features). Standard PCA uses linear projection to reduce the features. If the data is
# linearly separable (i.e., you can draw a straight line or hyperplane between different
# classes) then PCA works well. However, if your data is not linearly separable (e.g., you
# can only separate classes using a curved decision boundary), the linear transforma‐
# tion will not work as well.
# Kernels allow us to project the linearly inseparable data into a higher dimension
# where it is linearly separable; this is called the kernel trick.
# A common kernel to use is the Gaussian radial basis function kernel rbf , but
# other options are the polynomial kernel ( poly ) and sigmoid kernel (sigmoid).

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
features = StandardScaler().fit_transform(digits.data)
# Create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True) # whiten => transforms the values of each principal component so that they have mean = 0 and variance = 1
features_pca = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])


def get_pca(df):
    dates = df['Date']
    df.drop(columns=['Date'],inplace=True)
    x = df.values
    x = StandardScaler().fit_transform(x)

    n_components = params['pca_comp']
    pca_features = PCA(n_components=n_components)
    pcs = ['pc_'+str(el+1) for el in range(n_components)]
    principalComponents_features = pca_features.fit_transform(x)
    print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
        
    principal_features_df = pd.DataFrame(data=principalComponents_features,columns=pcs)
    df_res = pd.DataFrame()
    df_res['Date'] = dates
    df_res[pcs] = principal_features_df[pcs]
    # df_res[params['label']] = df[params['label']]
    return(df_res)

###############################
# (49) reducce featrues by maximizing class separability
###############################
# in PCA we were only interested in the component axes that maximize the variance
# in the data, while in LDA we have the additional goal of maximizing the
# differences between classes.

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
features = iris.data
target = iris.target

lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

###############################
# (50) feature selection bby variance thresholding
###############################

# Variance thresholding (VT) is one of the most basic approaches to feature selection. It
# is motivated by the idea that features with low variance are likely less interesting (and
# useful) than features with high variance.

from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

iris = datasets.load_iris()
features = iris.data
target = iris.target
thresholder = VarianceThreshold(threshold=.5)
features_high_variance = thresholder.fit_transform(features)

###############################
# (51) handling highly correlated features matrix => use correlation matrix
###############################

dataframe = pd.DataFrame(features)
corr_matrix = dataframe.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
dataframe.drop(dataframe.columns[to_drop], axis=1)

###############################
# (52) pipeline model
###############################

# Create features matrix
features = digits.data

# Create target vector
target = digits.target

# Create standardizer
standardizer = StandardScaler()

# Create logistic regression object
logit = LogisticRegression()

# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(pipeline, # Pipeline
                             features, # Feature matrix
                             target, # Target vector
                             cv=kf, # Cross-validation technique
                             scoring="accuracy", # Loss function
                             n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()

###############################
# (53) k-fold cross validation (KFCV)
###############################
# In KFCV, we split the data into k parts called “folds.” The model is then
# trained using k – 1 folds—combined into one training set—and then the last fold is
# used as a test set. We repeat this k times, each time using a different fold as the test
# set. The performance on the model for each of the k iterations is then averaged to
# produce an overall measurement.

###############################
# (54) create baseline regression model
###############################
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
# Load data
boston = load_boston()
# Create features
features, target = boston.data, boston.target
# Make test and training split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)
# Create a dummy regressor
dummy = DummyRegressor(strategy='mean')
# "Train" dummy regressor
dummy.fit(features_train, target_train)
# Get R-squared score
dummy.score(features_test, target_test)

# compare with simple linear regression

from sklearn.linear_model import LinearRegression
# Train simple linear regression model
ols = LinearRegression()
ols.fit(features_train, target_train)
# Get R-squared score
ols.score(features_test, target_test)

###############################
# (55) performance binary classifier
###############################

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# • TP is the number of true positives. Observations that are part of the positive class
# (has the disease, purchased the product, etc.) and that we predicted correctly.
# • TN is the number of true negatives. Observations that are part of the negative
# class (does not have the disease, did not purchase the product, etc.) and that we
# predicted correctly.
# • FP is the number of false positives. Also called a Type I error. Observations pre‐
# dicted to be part of the positive class that are actually part of the negative class.
# • FN is the number of false negatives. Also called a Type II error. Observations pre‐
# dicted to be part of the negative class that are actually part of the positive class.

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)

# F_1 = 2 × (Precision × Recall) / (Precision + Recall) # harmonic mean

logit = LogisticRegression()
# Cross-validate model using accuracy
cross_val_score(logit, X, y, scoring="accuracy") # or precision or recall or f1

###############################
# (56) confusion matrix
###############################

# Create training and test set
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)

# Create logistic regression
classifier = LogisticRegression()
# Train model and make predictions
target_predicted = classifier.fit(features_train,target_train).predict(features_test)
# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

###############################
# (57) simple linear regression
###############################

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 50,
                                   coef = False,
                                   random_state = 1)

# Create a linear regression object
ols = LinearRegression()
# Cross-validate the linear regression using (negative) MSE
cross_val_score(ols, features, target, scoring='neg_mean_squared_error') # or r2

###############################
# (58) clustering model
###############################
# Generate feature matrix
features, _ = make_blobs(n_samples = 1000,
                         n_features = 10,
                         centers = 2,
                         cluster_std = 0.5,
                         shuffle = True,
                         random_state = 1)

# Cluster data using k-means to predict classes
model = KMeans(n_clusters=2, random_state=1).fit(features)
# Get predicted classes
target_predicted = model.labels_
# Evaluate model
silhouette_score(features, target_predicted)

# While we cannot evaluate predictions versus true values if we don’t have a target vec‐
# tor, we can evaluate the nature of the clusters themselves. Intuitively, we can imagine
# “good” clusters having very small distances between observations in the same cluster
# (i.e., dense clusters) and large distances between the different clusters (i.e., well-
# separated clusters). Silhouette coefficients provide a single value measuring both
# traits. Formally, the ith observation’s silhouette coefficient is:

# s_i = (b_i- a_i) / max(a_i,b_i)

# where s_i is the silhouette coefficient for observation i, a_i is the mean distance between
# i and all observations of the same class, and b_i is the mean distance between i and all
# observations from the closest cluster of a different class. The value returned by sil
# houette_score is the mean silhouette coefficient for all observations. Silhouette coef‐
# ficients range between –1 and 1, with 1 indicating dense, well-separated clusters.

###############################
# (59) model selection + preprocessing in one step
###############################

# Create a preprocessing object that includes StandardScaler features and PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# Create a pipeline
pipe = Pipeline([("preprocess", preprocess),
("classifier", LogisticRegression())])

# Create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
"classifier__penalty": ["l1", "l2"],
"classifier__C": np.logspace(0, 4, 10)}]

# Create grid search
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# Fit grid search
best_model = clf.fit(features, target)

# View best model
best_model.best_estimator_.get_params()['preprocess__pca__n_components']

###############################
# (60) polynomial regression
###############################

# Create polynomial features x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_polynomial, target)

###############################
# (61) regularization
###############################

# In standard linear regression the model trains to minimize the sum of squared error
# between the true (y_i) and prediction, (y_i) target values, or residual sum of squares
# (RSS):
 
# RSS = ∑(y_i − \hat{y}_i)^2

# Regularized regression learners are similar, except they attempt to minimize RSS and
# some penalty for the total size of the coefficient values, called a shrinkage penalty
# because it attempts to “shrink” the model. There are two common types of regular‐
# ized learners for linear regression: ridge regression and the lasso. The only formal dif‐
# ference is the type of shrinkage penalty used. In ridge regression, the shrinkage pen‐
# alty is a tuning hyperparameter multiplied by the squared sum of all coefficients:

# RSS + \alpha ∑_{j=1,p} (\hat{β}_j)^2

# where β_j is the coefficient of the jth of p features and \alpha is a
# hyperparameter The lasso is similar, except the shrinkage penalty is a tuning
# hyperparameter multiplied by the sum of the absolute value of all coefficients:
# (1/2n) * RSS + α ∑_{j=1,p} |\hat{β}_j|

# in sklearn C is the inverse of the regularization strength: C = 1 / \alpha

###############################
# (62) tree models
###############################

# Tree-based learning algorithms are a broad and popular family of related non-
# parametric, supervised methods for both classification and regression. The basis of
# tree-based learners is the decision tree wherein a series of decision rules (e.g., “If their
# gender is male...”) are chained. The result looks vaguely like an upside-down tree,
# with the first decision rule at the top and subsequent decision rules spreading out
# below. In a decision tree, every decision rule occurs at a decision node, with the rule
# creating branches leading to new nodes. A branch without a decision rule at the end
# is called a leaf.

from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features, target)
observation = [[0.02, 16]]
model.predict(observation)

# G(t) = 1 - ∑_{i=1,c} (p_i)^2

# the Gini impurity at node t (G(t)) and p_i is the proportion of observations of
# class c at node t. This process of finding the decision rules that create splits to increase
# impurity is repeated recursively until all leaf nodes are pure (i.e., contain only one
# class) or some arbitrary cut-off is reached.

from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeRegressor(criterion='mse',random_state=0) # decision crierion based on MSE by default
model = decisiontree.fit(features, target)

# create graph of the decision tree => create DOT data
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)
# draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# show graph
Image(graph.create_png())
# Create PDF
graph.write_pdf("iris.pdf")


###############################
# (63) random forest
###############################
from sklearn.ensemble import RandomForestClassifier

# Create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
# Calculate feature importances
importances = model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [iris.feature_names[i] for i in indices]
# Create plot
plt.figure()
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.bar(range(features.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(features.shape[1]), names, rotation=90)
# Show plot
plt.show()

# In random forest, an ensemble (group) of randomized decision trees predicts the tar‐
# get vector. An alternative, and often more powerful, approach is called boosting. In
# one form of boosting called AdaBoost, we iteratively train a series of weak models
# (most often a shallow decision tree, sometimes called a stump), each iteration giving
# higher priority to observations the previous model predicted incorrectly. More specif‐
# ically, in AdaBoost:
# 1
# 1. Assign every observation, x_i , an initial weight value, w_i = 1/n , where n is the total
# number of observations in the data.
# 2. Train a “weak” model on the data.
# 3. For each observation:
# a. If weak model predicts x_i correctly, w_i is increased.
# b. If weak model predicts x_i incorrectly, w_i is decreased.
# 4. Train a new weak model where observations with greater w_i are given greater pri‐
# ority.
# 5. Repeat steps 4 and 5 until the data is perfectly predicted or a preset number of
# weak models has been trained.
#
# The end result is an aggregated model where individual weak models focus on more
# difficult (from a prediction perspective) observations. In scikit-learn, we can imple‐
# ment AdaBoost using AdaBoostClassifier or AdaBoostRegressor . The most impor‐
# tant parameters are base_estimator , n_estimators , and learning_rate :
# • base_estimator is the learning algorithm to use to train the weak models. This
# will almost always not need to be changed because by far the most common
# learner to use with AdaBoost is a decision tree—the parameter’s default argu‐
# ment.
# • n_estimators is the number of models to iteratively train.
# • learning_rate is the contribution of each model to the weights and defaults to 1 .
# Reducing the learning rate will mean the weights will be increased or decreased

###############################
# (64) logistic regression
###############################

# Despite being called a regression, logistic regression is actually a widely used super‐
# vised classification technique. Logistic regression and its extensions, like multinomial
# logistic regression, allow us to predict the probability that an observation is of a cer‐
# tain class

# Despite having “regression” in its name, a logistic regression is actually a widely used
# binary classifier (i.e., the target vector can only take two values). In a logistic regres‐
# sion, a linear model (e.g., β_0 + β_1 x) is included in a logistic (also called sigmoid) function, such that:

# P(y_i = 1 | X = 1) = 1 / (1+e^[-(β_0 + β_1 x)])

# P(y_i = 1 | X = 1) is the probability of the ith observation’s target value, y i , being class
# 1, X is the training data, β 0 and β 1 are the parameters to be learned, and e is Euler’s
# number. The effect of the logistic function is to constrain the value of the function’s
# output to between 0 and 1 so that it can be interpreted as a probability. If P(y i = 1 | X)
# is greater than 0.5, class 1 is predicted; otherwise, class 0 is predicted.

###############################
# (65) SVC/SVM
###############################

# SVC attempts to find the hyperplane—a line when we only have two
# dimensions—with the maximum margin between the classes.

# use if SVC with imbalanced data => add class_weight to be balanced
# change the kernel from linear to rbf or other

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create support vector classifier
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
# Train classifier
model = svc.fit(features_standardized, target)

###############################
# (66) Naive Bayes
###############################

P(y | x_1,..,x_j) =  P(x_1,..,x_j | y) * P(y) / P(x_1,..,x_j)

• P(y | x_1,..,x_j ) is called the posterior and is the probability that an observation is
class y given the observation’s values for the j features, x_1,..,x_j.
    
• P(x_1,..,x_j | y) is called likelihood and is the likelihood of an observation’s values for
features, x 1 , ..., x j , given their class, y.

• P(y) is called the prior and is our belief for the probability of class y before look‐
ing at the data.

• P(x_1,..,x_j) is called the marginal probability.

# For each feature in the data, we have to assume the statistical distribution of the likelihood,
# P(x_j | y). The common distributions are the normal (Gaussian), multinomial, and
# Bernoulli distributions. The distribution chosen is often determined by the nature of
# features (continuous, binary, etc.). Second, naive Bayes gets its name because we
# assume that each feature, and its resulting likelihood, is independent.

from sklearn.naive_bayes import GaussianNB

classifer = GaussianNB()
model = classifer.fit(features, target)
new_observation = [[ 4, 4, 4, 0.4]]
model.predict(new_observation)

###############################
# (67) k-means clustering
###############################

# In k-means clustering, the algorithm attempts to group observations into k groups, with each
# group having roughly equal variance. The number of groups, k, is specified by the
# user as a hyperparameter:

# 1. k cluster “center” points are created at random locations.
# 2. For each observation:
# a. The distance between each observation and the k center points is calculated.
# b. The observation is assigned to the cluster of the nearest center point.
# 3. The center points are moved to the means (i.e., centers) of their respective
# clusters.
# 4. Steps 2 and 3 are repeated until no observation changes in cluster membership.

from sklearn.cluster import KMeans

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)
model = cluster.fit(features_std)
# View predicted classes
model.labels_

###############################
# (68) clustering w/o knowing the number of clusters => MeanShift
###############################
from sklearn.cluster import MeanShift

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = MeanShift(n_jobs=-1)
model = cluster.fit(features_std)

###############################
# (69) clustering w/o knowing the number of clusters => DBSCAN
###############################

# DBSCAN is motivated by the idea that clusters will be areas where many observations
# are densely packed together and makes no assumptions of cluster shape. Specifically,
# in DBSCAN:

# 1. A random observation, x_i , is chosen.
# 2. If x_i has a minimum number of close neighbors, we consider it to be part of a
# cluster.
# 3. Step 2 is repeated recursively for all of x_i ’s neighbors, then neighbor’s neighbor,
# and so on. These are the cluster’s core observations.
# 4. Once step 3 runs out of nearby observations, a new random point is chosen (i.e.,
# restarting step 1).

# Once this is complete, we have a set of core observations for a number of clusters.
# Finally, any observation close to a cluster but not a core sample is considered part of a
# cluster, while any observation not close to the cluster is labeled an outlier.

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = DBSCAN(n_jobs=-1)
model = cluster.fit(features_std)

###############################
# (70) agglomerative clustering 
###############################

# Agglomerative clustering is a powerful, flexible hierarchical clustering algorithm. In
# agglomerative clustering, all observations start as their own clusters. Next, clusters
# meeting some criteria are merged together. This process is repeated, growing clusters

# until some end point is reached. In scikit-learn, AgglomerativeClustering uses the
# linkage parameter to determine the merging strategy to minimize the following:

# 1. Variance of merged clusters ( ward )
# 2. Average distance between observations from pairs of clusters ( average )
# 3. Maximum distance between observations from pairs of clusters ( complete )

# Two other parameters are useful to know. First, the affinity parameter determines
# the distance metric used for linkage ( minkowski , euclidean , etc.). Second, n_clus
# ters sets the number of clusters the clustering algorithm will attempt to find. That is,
# clusters are successively merged until there are only n_clusters remaining.

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = AgglomerativeClustering(n_clusters=3)
model = cluster.fit(features_std)


###############################
# (71) Neural Networks
###############################

from sklearn import preprocessing
import numpy as np

# features and scaling
features = np.random.normal(0.0,1.0,[5,2]) # 10 elements
scaler = preprocessing.StandardScaler()
features_standardized = scaler.fit_transform(features)

# Load libraries
from keras import models
from keras import layers
# Start neural network
network = models.Sequential()
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))
# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))
# Compile neural network
network.compile(loss="binary_crossentropy", # Cross-entropy
optimizer="rmsprop", # Root Mean Square Propagation
metrics=["accuracy"]) # Accuracy performance metric

###############################
# (72) reduce overfitting by early stopping
###############################

# When our neural network is new, it will have a poor performance. As the neural net‐
# work learns on the training data, the model’s error on both the training and test set
# will tend to increase. However, at a certain point the neural network starts “memoriz‐
# ing” the training data, and overfits. When this starts happening, the training error
# will decrease while the test error will start increasing. Therefore, in many cases there
# is a “sweet spot” where the test error (which is the error we mainly care about) is at its
# lowest point.

# In the first training epochs both the training and test errors will decrease,
# but at some point the network will start “memorizing” the training data,
# causing the training error to continue to decrease even while the test error
# starts increasing. Because of this phenomenon, one of the most common and very
# effective methods to counter overfitting is to monitor the training process and
# stop training when the test error starts to increase. This strategy is called
# early stopping.

# Load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Set random seed
np.random.seed(0)
# Set the number of features we want
number_of_features = 1000
# Load data and target vector from movie review data
(data_train, target_train), (data_test, target_test) = imdb.load_data(
num_words=number_of_features)
# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Start neural network

network = models.Sequential()
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16,
activation="relu",
input_shape=(number_of_features,)))
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))
# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))
# Compile neural network
network.compile(loss="binary_crossentropy", # Cross-entropy
optimizer="rmsprop", # Root Mean Square Propagation
metrics=["accuracy"]) # Accuracy performance metric
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor="val_loss", patience=2),
ModelCheckpoint(filepath="best_model.h5",
monitor="val_loss",
save_best_only=True)]
# Train neural network
history = network.fit(features_train, # Features
                      target_train, # Target vector
                      epochs=20, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=0, # Print description after each epoch
                      batch_size=100, # Number of observations per batch
                      validation_data=(features_test, target_test)) # Test data


###############################
# reduce overfitting by adding dropout
###############################
# Dropout is a popular and powerful method for regularizing neural networks. In
# dropout, every time a batch of observations is created for training, a proportion of
# the units in one or more layers is multiplied by zero (i.e., dropped). In this setting,
# every batch is trained on the same network (e.g., the same parameters), but each
# batch is confronted by a slightly different version of that network’s architecture.

# Dropout is effective because by constantly and randomly dropping units in each
# batch, it forces units to learn parameter values able to perform under a wide variety
# of network architectures. That is, they learn to be robust to disruptions (i.e., noise) in
# the other hidden units, and this prevents the network from simply memorizing the
# training data.
# It is possible to add dropout to both the hidden and input layers. When an input layer
# is dropped, its feature value is not introduced into the network for that batch. A com‐
# mon choice for the portion of units to drop is 0.2 for input units and 0.5 for hidden
# units.

# Load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
# Set random seed
np.random.seed(0)
# Set the number of features we want
number_of_features = 1000
# Load data and target vector from movie review data
(data_train, target_train), (data_test, target_test) = imdb.load_data(
num_words=number_of_features)
# Convert movie review data to a one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Start neural network
network = models.Sequential()
# Add a dropout layer for input layer
network.add(layers.Dropout(0.2, input_shape=(number_of_features,)))
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))
# Add a dropout layer for previous hidden layer
network.add(layers.Dropout(0.5))
# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))
# Add a dropout layer for previous hidden layer
network.add(layers.Dropout(0.5))
# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))
# Compile neural network
network.compile(loss="binary_crossentropy", # Cross-entropy
optimizer="rmsprop", # Root Mean Square Propagation
metrics=["accuracy"]) # Accuracy performance metric
# Train neural network
history = network.fit(features_train, # Features
                      target_train, # Target vector
                      epochs=3, # Number of epochs
                      verbose=0, # No output
                      batch_size=100, # Number of observations per batch
                      validation_data=(features_test, target_test)) # Test data
