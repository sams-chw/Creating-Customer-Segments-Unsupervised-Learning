#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
import copy
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data_copy = copy.deepcopy(data)
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
print("\n",data.head())


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:


# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:


# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [23, 105, 437]
# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 3071.9
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

# In[4]:


stat = pd.concat([np.round(data.mean(), 1), data.median()], axis = 1).T
stat.index = ['mean', 'median']
print(stat)

for k,i in enumerate(indices):
    print("\nStatistical comparison for sample %d (index=%d)" % (i,k))
    print("===============================================",)
    stat_sample = pd.concat([np.round(data.iloc[i] - data.mean(), 1), data.iloc[i] - data.median()], axis = 1)
    stat_sample.columns = ['(X - mean)', '(X - median)']
    print(stat_sample)


# **Answer:**
# 1.	For sample index 23, this establishment spends significantly more than the mean and median in all categories (even greater than 75% percentile in most cases), which indicates that they could be a large retailer.
# 2.	For sample index 105, this establishment spends more than average on fresh items and very close to average on delicatessen items, whereas they spend significantly less than average in all the other categories (milk, grocery, frozen and detergents_paper), particularly detergents_paper items. This indicates that this establishment could be a deli with salad/sandwich shop OR a small fresh food grocery store.
# 3.	For sample index 437, this establishment spends more than the mean and median in all categories except frozen items, which indicates that they could be a big hotel/restaurant.

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
print(data.keys())
feature_to_drop = 'Fresh'
new_data = data.drop([feature_to_drop], axis=1)

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
X_train, X_test, y_train, y_test = train_test_split(new_data, data[feature_to_drop], test_size=0.25, random_state=1)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train,y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)
print ("Score:", score)


# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to beleive that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.

# **Answer:**
# I attempted to predict the "Fresh" feature using the other features. The prediction score was -0.9234 indicating that there is no correlation between the "Fresh" and the other features. Therefore, "Fresh" is necessary to include in the data to identify customer's spending habits.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[6]:


# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[7]:


import seaborn as sns

sns.heatmap(data.corr(), annot=True)


# In[8]:


sns.pairplot(data)


# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html) (the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

# In[9]:


from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt

for i in range(len(data.columns)):
    qqplot(data.iloc[:,i], line='s')
    plt.title('Q-Q plot: %s' % data.columns[i])
    plt.show()


# **Answer:**
# Data is not normally distributed (see the Q-Q plots above) and in almost all cases, the mean and median vary significantly which indicates the existence of a large skew. The presence of outliers can also contribute to the skewness of the data. 
# 
# As seen in the scatter matrix, there are several pairs of features that show strong linear correlation. For example, milk-grocery (0.73), milk-detergents_paper (0.66), grocery-detergents_paper (0.92), etc.
# 
# In the previous question, the prediction score was found to be -0.9234 when attempting to predict ‘Fresh’ feature using the other features which indicates the relevance of this feature in identifying a specific customer. From the above heatmap/scatter matrix, it is found that the ‘Fresh’ feature has very little or no correlation with other features which consolidates my earlier finding. 
# 
# Looking at its density plot, the "Fresh" feature is heavily skewed towards the origin and does not follow a normal profile. The other features which show high correlation with other features do not provide much information gain and hence can be considered irrelevant in identifying a specific customer.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[10]:


# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[11]:


# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[27]:


# For each feature find the data points with extreme high or low values
outliers = []
repeat_outliers = []
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)
    print("Data points considered outliers for the feature '{}':".format(feature))
    feature_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(feature_outliers)

    # append outlier indices to outliers array
    for i in feature_outliers.index:
        if i not in outliers:
            outliers.append(i)
        elif i not in repeat_outliers:
            repeat_outliers.append(i)

print("Number of outliers: ", len(outliers))
print("Outliers: ", outliers)
print("")
print("Number of repeat outliers: ", len(repeat_outliers))
print("Repeat outliers: ", repeat_outliers)
# Remove the outliers, if any were specified
print(log_data.shape)
good_data = log_data.drop(log_data.index[outliers])
old_index = good_data.index.values
good_data = good_data.reset_index(drop=True)
good_data_copy = copy.deepcopy(good_data)
good_data_copy['Old_index'] = old_index


# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.

# **Answer:**
# There are 5 data points that are outliers for more than one feature. The indices of the repeat outliers are [154, 65, 75, 66, 128].
# 
# All outliers that are outside the acceptable range ( Q1-1.5*IQR < x < Q3 + 1.5 * IQR ) should be eliminated to prevent them from imposing a skewing effect on the profile of the rest of the data.
# 
# In the case of k-means clustering, if the centroids are chosen to be the centers of the true clusters (the best 'representative' configuration), the value of the loss function can be very high due to the existence of outliers far from the nearest centroid. Therefore, k-means algorithm will reduce the loss function by pushing the cluster center closer to the outlier. The resulting configuration will clearly not be the representative of the underlying distribution.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[13]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
n_components = 6
pca = PCA(n_components=n_components).fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:**
# The first and second principal components account for 72.5% of the explained variance in the data.
# 
# The first four principal components account for 92.8% of the explained variance in the data.
# 
# The first principal component (PC) primarily represents variation in spending on "Detergents_Paper", and provides information gain for "Grocery" and "Milk" (less positive weights than that of "Detergents_Paper"). The positive direction on the first component indicates higher spending on these product categories. It should be noted here that “Detergents_Paper” exhibits high correlation with “Grocery” and “Milk” as previously found.
# 
# The second principal component largely represents variation (information gain) in spending on "Fresh" and "Frozen", and to a slightly lesser extent, "Delicatessen". It should be noted here that "Fresh" and "Frozen" exhibit very little or no correlation with other features. In the case of second PC, all features (except "Detergents_Paper" which has a very small negative weight) are positively weighted meaning that an increase in spending in any of these features will lead to an overall increase in the PC.
# 
# Variation in the third principal component is driven largely by variation in 'Fresh' and 'Delicatessen' spending. However, the former is weighted negatively, meaning an increase in 'Fresh' product spending will cause a decrease in the principal component. The component is also sensitive to variation in 'Frozen' and 'Detergents_Paper' spending (inversely in the case of the latter).
# 
# The fourth principal component is strongly negatively weighted towards 'Frozen' product spending, and positively weighted towards 'Delicatessen' product spending. It is moderately (negatively) influenced by 'Detergents_Paper' spending and (positively) influenced by 'Fresh' product spending.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[14]:


# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[15]:


# TODO: Apply PCA by fitting the good data with only two dimensions
n_components = 2
pca = PCA(n_components=n_components).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
vs.pca_results(good_data, pca)


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[16]:


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[17]:


# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.

# **Answer:**
# 
# Hard clustering describes a family of algorithms where each sample in a dataset is assigned to exactly one cluster, whereas algorithms for soft clustering assign a sample to one or more clusters.
# 
# **K-Means** is a hard clustering algorithm, in which points are explicitly assigned to one of k clusters. The main advantage of K-Means clustering is its computational efficiency which allows for scalability to very large data sets. The disadvantage of K-Means is that it assumes the clusters are independent and spherical (thus having equal variance), which doesn't allow for more complex geometries within the data (e.g. non-linear clusters).
# 
# **Gaussian Mixture Model (GMM),** on the other hand, is a soft clustering algorithm. They produce probabilistic models that assume all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. In fact, K-Means algorithm is essentially a special case of GMM with equal covariance per component. The main advantage of GMM clustering is that it doesn't assume the features in a cluster having same variance, and thus supports for non-spherical clusters, e.g., ellipsoids. The disadvantage to Gaussian Mixture Model clustering is that when one has insufficiently many points per mixture, estimating the covariance matrices becomes difficult, and the algorithm is known to diverge and find solutions with infinite likelihood unless one regularizes the covariances artificially (source: https://scikit-learn.org/stable/modules/mixture.html).
# 
# 
# **Choice of algorithm:** GMM clustering algorithm will be used for this problem. 
# 
# Given the lack of prior knowledge around the different customer types (i.e. clusters) represented by the dataset, the additional flexibility in assumptions offered by GMM over K-Means makes this model a better first choice.

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[18]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

n_clusters = 2

# TODO: Apply your clustering algorithm of choice to the reduced data 
clusterer = GaussianMixture(n_components=n_clusters).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)
print(reduced_data.shape)

# TODO: Find the cluster centers
centers = clusterer.means_
print(centers)

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)
print(sample_preds)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)

print("Score for", n_clusters, "clusters:", np.round(score, 4))


# In[19]:


def generate_cluster_score(model, n, data):
    n_clusters_ = n
    if model == 'GaussianMixture':
        from sklearn.mixture import GaussianMixture
        clusterer_ = GaussianMixture(n_clusters_, random_state=51).fit(data)
    elif model == 'KMean':
        from sklearn.cluster import KMeans
        clusterer_ = KMeans(n_clusters_, random_state=29).fit(data)
    preds_ = clusterer_.predict(data)
    return silhouette_score(data, preds_)


# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?

# In[20]:


score_array = []
model_name = ['GaussianMixture', 'KMean']

for name in model_name:
    for i in range(2,21):
        score_array.append(generate_cluster_score(name, i, reduced_data))

    print("\n*** Scores for %s model ***" % name)
    print("{0:10s} {1:13s} {2:6s}".format("#Clusters", " ", "Score"))
    print("===============================")
    for i in range(len(score_array)):
        print("{0:5d} {1:18s} {2:5.3f}".format(i + 2, " ", round(score_array[i], 4)))


# **Answer:** 
# The silhouette scores for several numbers of clusters for both K-Means and GMM clustering have been populated above.
# 
# The silhouette score varies with number of clusters, generally decreases as the number of clusters increases. Both in the case of K-Means and GMM clustering, the best silhouette scores were obtained using 2 clusters.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[21]:


# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[22]:


# TODO: Inverse transform the centers
print("centers")
print(centers.shape)
print(centers)

print("log centers:")
log_centers = pca.inverse_transform(centers)
print(log_centers.shape)
print(log_centers)

# TODO: Exponentiate the centers
print("true centers:")
true_centers = np.exp(log_centers)
print(true_centers.shape)
print(true_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
print(segments)
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
print(true_centers)
true_centers.index = segments
display(true_centers)


# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

# In[23]:


# print(true_centers.iloc[0,:])
# print(data.mean())
# print(data.std(ddof=0))

for i in range(0,len(centers)):
    print('Statistical comparison for Segment_{} data'.format(i))
    df0 = pd.DataFrame([true_centers.iloc[i,:], data.mean(),
                        (true_centers.iloc[i,:] - data.mean())/data.std(ddof=0),
                        (true_centers.iloc[i,:] - data.describe().iloc[4,:])/data.std(ddof=0),
                        (true_centers.iloc[i,:] - data.median())/data.std(ddof=0),
                        (true_centers.iloc[i,:] - data.describe().iloc[6,:])/data.std(ddof=0)],
                        index=['True_centers', 'Mean', '^mean', '^25%tile', '^50%tile', '^75%tile'])
    print(np.round(df0,3).T)
    print()
    
print("where '^' indicates values(Xdata) measured as (true_centers - Xdata)/StandardDeviation(Xdata)")


print("Heatmap for w.r.t. mean:")
sns.heatmap((true_centers-data.mean())/data.std(ddof=0),
            square=True, annot=True, cbar=False)
plt.show()


# **Answer:**
# 
# 
# **Segment 0:** The average customer assigned to this cluster has a below average spending in all product categories, with particularly low 'Detergents_Paper' and 'Delicatessen' spending. This customer spends the most (although less than mean value)  on 'Fresh' products and a reasonable amount (but much less than average) on 'Milk', 'Grocery' or 'Frozen' products, which are evenly balanced. Notably, spending on 'Fresh' and 'Frozen' products is greater than the median but less than mean values for this category. Given this spending profile, this customer could be a restaurant.
# 
# **Segment 1:** The average spending for customers assigned to this cluster on 'Milk', 'Grocery' and 'Detergents_Paper' products is much more than their mean values (even greater than their 75%tile) for this category. This customer spends a reasonable amount (although less than average and median) on 'Fresh' products. This customer does not spend a lot on 'Frozen' or 'Delicatessen' products. Given this spending profile, this customer could be a supermarket or large convenience store.
# 
# 

# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[24]:


# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
    
print()

for i in range(0,len(samples)):
    print('Statistical comparison for sample_{} data'.format(i))
    df0 = pd.DataFrame([samples.iloc[i,:], data.mean(),
                        (samples.iloc[i,:] - data.mean())/data.std(ddof=0),
                        (samples.iloc[i,:] - data.describe().iloc[4,:])/data.std(ddof=0),
                        (samples.iloc[i,:] - data.median())/data.std(ddof=0),
                        (samples.iloc[i,:] - data.describe().iloc[6,:])/data.std(ddof=0)],
                        index=['True_centers', 'Mean', '^mean', '^25%tile', '^50%tile', '^75%tile'])
    print(np.round(df0,3).T)
    print()


# **Answer:** The customer represted by sample 0 spends highly (even higher than 3rd quartile) in all categories and thus the learning algorithm has correctly predicted this customer to be belonging to cluster 1.
# 
# In the case of sample 1, the spending on all categories (except fresh products, which is in between mean and 3rd quartile) is much less than average, particularly spending on frozen and detergent_paper products (which is close to or less than 1st quartile) is notable. Therefore, this customer should belong to cluster 0.
# 
# For sample 2, the spending on milk, grocery and detergent_paper products is quite high (higher than 3rd quartile). This customer spends a reasonable amount (but more than average) on fresh and delicatessen items but interestingly quite less (even less than 1st quartile) on frozen items. Since the spending on most categories is above average, this cutomemr should belong to cluster 1. 

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:**
# The model has established two main customer types - Cluster 1 representing 'supermarkets'/'bulk distributors' (who stock lots of different items) and Cluster 0 representing 'restaurants/cafes' who stock fresh food. It is likely that customers from Cluster 0 who serve lots of fresh food everyday would prefer the 5 days a week delivery service in order to keep food as fresh as possible. Cluster 1 could be more flexible - they buy a more wide variety of perishable and non-perishable goods so do not necessarily need a daily delivery.
# 
# Therefore, it cannot be assumed that the change will affect all customers equally. The A/B test should be carried out on representative samples of customers from each segment to determine how the change affects both groups differently and then evaluate feedback separately.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:**
# We can train a supervised learning algorithm (such as support vector machine) on the customer data, using annual spending on each product category as our features, and customer segment as our labels. We can then use this trained classifier to predict the customer segment for each new customer, and assign each one a delivery frequency based on their assigned customer segment.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[25]:


# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# In[26]:


data_copy = data_copy[data_copy.index.map(lambda x: x in old_index).values]
data_copy['Predicted_channel'] = preds+1
data_copy['Compare_channel'] = data_copy['Channel'] == data_copy['Predicted_channel']
print(data_copy.head())
print(data_copy['Compare_channel'].value_counts())
accuracy = data_copy['Compare_channel'].value_counts().iloc[0]/data_copy.shape[0]*100
print('Accuracy = {}%'.format(round(accuracy,3)))


# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?

# **Answer:**
# 
# The predicted clusters match very closey (with an accuracy of 89.7%) with the actual data and the classification (i.e.,  cluster 0 aligns with the "Retailer" feature, while cluster 1 aligns with the "Hotel/Restaurant/Cafe" feature). However, there are still several 'HoReCa' ('Retail') customers that the algorithm incorrectly identifies as 'Retail' ('HoReCa') by placing them in Cluster 1 (Cluster 0). 
# 
# The actual distribution has a less well defined seperation between clusters (as compared to the predicted ones). However, it can be stated with a reasonable confidence that datapoints that lie in the area of a very positive 1st PC (>3.5) and 2nd PC (>1) are most certainly Retailers. On the other hand, data points with a very negative 1st PC (<-3) and 2nd PC (<-1) are most certainly Hotels/Restaurants/Cafes.
# 
# Yes, the actual classification is very close to the guesses I made regarding each customer segment (see answer to question 8), i.e., Cluster 0 to be Restaurants (I didn't consider hotels/cafes) and Cluster 1 to supermarkets or large convenience stores (analagous to retailers). 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
