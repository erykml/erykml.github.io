---
title: "Ensemble learning using the Voting Classifier"
categories: 
  - Data Science
last_modified_at: 2020-02-09
---

<img class="ds t u eq ak" src="https://miro.medium.com/max/3840/1\*jRPs81mYrrmMsgVP9suJrA.jpeg" width="1920" height="1282" role="presentation"/>

Ensemble learning using the Voting Classifier
=============================================

Learn how to leverage the strengths of multiple models using a variant of ensemble learning
-------------------------------------------------------------------------------------------

[

In this article, I describe a simple ensemble algorithm. In general, ensemble models combine multiple base models to improve the predicting performance. The best-known example of an ensemble model is the Random Forest, which — greatly simplifying the algorithm’s logic — combines multiple Decision Trees and aggregates their predictions using majority vote in case of a classification problem or by taking the average for regression tasks.

Similarly to the Random Forest, the Voting Ensemble estimates multiple base models and uses voting to combine the individual predictions to arrive at the final ones. However, the key difference lies in the base estimators. Models such as Voting Ensemble (and Stacking Ensemble) do not require the base models to be homogenous. In other words, we can train different base learners, for example, a Decision Tree and a Logistic Regression, and then use the Voting Ensemble to combine the results.

The following diagram presents the workflow of the Voting Ensemble:

<img class="ds t u eq ak" src="https://miro.medium.com/max/4896/1\*EekXroXi99N05jcskdPsIg.png" width="2448" height="828" role="presentation"/>

Voting Classifier supports two types of voting:

*   **hard**: the final class prediction is made by a majority vote — the estimator chooses the class prediction that occurs most frequently among the base models.
*   **soft**: the final class prediction is made based on the average probability calculated using all the base model predictions. For example, if model 1 predicts the positive class with 70% probability, model 2 with 90% probability, then the Voting Ensemble will calculate that there is an 80% chance of the observation belonging to the positive class and choose the positive class as the prediction. Additionally, we can use custom weights to calculate the weighted average. This is apt for cases in which we put more trust in some models, but still want to consider the ones we trust less.

One thing to bear in mind is that in order to use soft voting, all the base models must have the `predict_proba` method. Soft voting can result in better performance than hard voting (but not necessarily), as by averaging the probabilities it “gives more weight” to the confident votes.

The abovementioned voting schemes are only valid for a classification problem (both binary and multi-class). In `scikit-learn` there is a separate voting estimator for regression problems (`VotingRegressor`), which uses the average of the base estimators’ predictions as the final prediction. For an example of implementation, please see [the Notebook on GitHub](https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/voting_classifier_article.ipynb).

The Voting Ensemble is a useful technique, which comes especially handy when a single model shows some kind of bias. It is also possible that the Voting Ensemble results in a better overall score than the best of the base estimators, as it aggregates the predictions of multiples models and tries to cover for potential weaknesses of the individual models. One way of improving the ensemble’s performance is by making the base estimators as diverse as possible.

With version 0.22, `scikit-learn` introduced the `StackingClassifier`, which is another ensemble technique that uses heterogeneous models as the base estimators. The difference between stacking and voting ensembles is that the former trains another estimator (called meta learner) on top of the base estimators’ predictions. A common model used for that task is Logistic Regression. It is also possible to stack multiple layers of models on top of each other.

After the brief theoretical introduction, let’s get our hands dirty with coding.

Setup
=====

We need to import the following libraries:

The list is quite lengthy, as we need to import different classifiers, which will be used as base estimators.

Preparing the dataset for classification
========================================

For simplicity, we prepare the classification dataset using `scikit-learn`’s `make_classification` function. We create a dataset with 500 observations, 10 features. Using the default settings, there are 2 informative features in the dataset and 2 redundant ones. For more information on the function, please refer to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html). We additionally set the `random_state` for reproducibility.

The dataset we created is balanced, with the ratio of label classes of 1:1. We can verify this by running `Counter(y)`.

After creating the dataset, we split the data into training and test sets (using the 80–20 split) and scale the features using StandardScaler. Remember that when applying such transformations as scaling, we train the scaler only on the training set and transform both the training and test sets. This way we prevent data leakage.

Algorithms such as Decision Trees do not require scaling the features, however, estimators that rely on some kind of a distance metric (such as k-Nearest Neighbours) do. We scaled the data so it is compatible with all the considered estimators.

Fitting the models
==================

As the first step, we define a list of tuples, each one containing the name of the model and the estimator itself. At this point, we are satisfied with the default settings of all the classifiers. We set up the models this way, as the list will be used as the input for the `VotingClassifier` later on.

In the next step, we iterate over the previously defined list. For each model, we fit it to the training data, predict the labels for the test set and evaluate the performance using accuracy (it is an acceptable metric here, as the classes are balanced). For reproducibility's sake, whenever an estimator has a `random_state` parameter, we set it to 42.

The following snippet summarizes the performance of the models:

```
decision tree's accuracy: 0.86   
logistic regression's accuracy: 0.85   
knn's accuracy: 0.86   
naive bayes classifier's accuracy: 0.87 
```

The best performing model is the Naive Bayes Classifier.

Using the VotingClassifier
==========================

It is time to see if we can improve the performance by using the `VotingClassifier`. We start with the hard voting scheme (majority vote). The following code present how to set it up:

Running the code prints the accuracy score of 0.88, which is slightly better than the best of the base estimators. It seems that this time we could benefit from using the ensemble approach.

We can also try to use the soft voting by replacing the value of the voting parameter with `'soft’`. Doing so actually reduces the accuracy score to 0.85. This could be caused by the fact that in some cases the models were not really confident about the predicted class.

Conclusions
===========

In this article, I showed a basic use case for one of the ensemble methods available in `scikit-learn`. When working on a project, we most likely experiment with different estimators either way, so it only makes sense to try to aggregate them and see if it leads to better results. A potential drawback of using such an approach in a real-life project could be the lack of interpretability.

Additionally, the `VotingClassifier` can be used together with `GridSearchCV` (or the randomized variant) to find the best hyperparameter values for the base estimators.

You can find the code used for this article on my [GitHub](https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/voting_classifier_article.ipynb). As always, any constructive feedback is welcome. You can reach out to me on [Twitter](https://twitter.com/erykml1?source=post_page---------------------------) or in the comments.

* * *

I recently published a book on using Python for solving practical tasks in the financial domain. If you are interested, I posted [an article](https://towardsdatascience.com/introducing-my-book-python-for-finance-cookbook-de219ca0d612) introducing the contents of the book. You can get the book on [Amazon](https://www.amazon.com/gp/product/1789618517/ref=as_li_qf_asin_il_tl?ie=UTF8&tag=erykml-20&creative=9325&linkCode=as2&creativeASIN=1789618517&linkId=51dc32b8f827bf696fd898d6071fe53e) (disclaimer: referral link) or [Packt’s website](https://bit.ly/2SbBNcj).

<img class="ds t u eq ak" src="https://miro.medium.com/max/488/1\*zPB7oDayV2o3vhxMRoAuUw.png" width="244" height="304" role="presentation"/>
