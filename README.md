# Starbucks

Starbucks is an American coffee company and coffeehouse chain. This report explores consumer behavior in relation to an undisclosed product. Information such as customer demographics, spending habits, and offer type are present in separate datasets. By manipulating these features we are able to segment customers, predict spending habits, and suggest offers to customers.

The complete report can be found [inside the repository](https://github.com/NadimKawwa/starbucks/blob/master/strabucks_report_capstone.pdf) or on [google drive](https://drive.google.com/file/d/1oF6u1BFvCpc17uXyLLz0dTD_9b0PIZoh/view?usp=sharing).

## System Requirements

The scripts are written in python 3.x, the following packages are required:

- numpy
- pandas
- sklearn
- seaborn
- scipy
- tqdm
- json

## Datasets

The data is contained in three files inside the data folder. Note that due to size they are zipped.

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


## Cleaning the Data

The [first notebook](https://github.com/NadimKawwa/starbucks/blob/master/00_Starbucks_Capstone_notebook_Cleaning.ipynb) deals with exploring and cleaning the data. We can infer certain relationships from the data such as for the portfolio in the correlation heatmap below:

![portfolio_corr](https://github.com/NadimKawwa/starbucks/blob/master/plots/portfolio_corr.png)

When determining if an amount spent was influenced by an offer we shall assume that:
- Influence period begins when an offer is seen
- Influence period ends whenever an offer expires or is completed, whichever comes first
- No two offers interact, the first one seen is the one that governs

The process is summarized in the table below for two sample offers O1 and O2:
![influence_assumption](https://github.com/NadimKawwa/starbucks/blob/master/plots/cleaning_assumption.png)


## Clustering Customers

In the [second notebook](https://github.com/NadimKawwa/starbucks/blob/master/01_Starbucks_Capstone_notebook_segmentation.ipynb) we attempt to see if we can reasonably cluster our customers. As a first step we use the elbow method to determine the number of clusters:
![mini_k_elbow](https://github.com/NadimKawwa/starbucks/blob/master/plots/elbow_minik.png)

The resulting clusters visualized are shown in the plot below. However with a silhouette score of 0.1, our clustering is not completely adequate.
![pca_space](https://github.com/NadimKawwa/starbucks/blob/master/plots/pca_space.png)


## Predicting Offers Seen

The [third notebook](https://github.com/NadimKawwa/starbucks/blob/master/02_Starbucks_Capstone_notebook_SupevisedLearning.ipynb) attempts to predict how many offers a user will see using supervised learning. The best performer in terms of accuracy, f1 score, and time complexity is support vector machines. The accuracy is around 51% and the average f1 score is 0.51.
The confusion matrix is shown below:

![svc_pipe_seen](https://github.com/NadimKawwa/starbucks/blob/master/plots/svc_pipe_seen_confusion.png)

## Creating a Recommendation Engine

In the [fourth](https://github.com/NadimKawwa/starbucks/blob/master/03_Starbucks_Capstone_notebook_RecommendationEngine.ipynb) and final notebook, we build a recommendation engine for the users using three methods:
- Most popular offer
- User-user similarity
- FunkSVD

For most popular offer we consider those that have the highest view ratio and those that yield the highest return per view.

Collaborative filtering with user-user similarity is done by using methods such as Euclidian distance and Pearson's correlation coefficient.


## Conclusion

Predicting consumer spending habits is not a trivial task, this report has made several assumptions about those habits. Namely that purchases that are seen influence spending and that no two offers interact.
We have seen how we can predict customers seeing an offer with supervised learning algorithms while keeping in mind time complexity.
Finally, the report offers a business application by proposing different methods to suggest offers to users.
