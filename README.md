<div align="center">
<h1> Predict Customer Response </h1>
 
 <p align="center">
<img src="https://github.com/raofida75/predict-customer-response/blob/main/image/cover.jpg" width="500"/>
</p>

<i> Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). In this project, I will try to build a model which can predict whether a user will complete the offer.
</i></div>

## Project Goals

- What is the average age of the Starbucks' customer?
- Which are the most successful offers?
- Determine the success of each offer by gender and age groups.
- Create a ML model to predict who is likely to complete the offer once he/she views the offer given the demographics of the user and offer characteristics.
- Which features will are the main drivers for predicting the user response.

## Datasets

The data is contained in three files:

- portfolio.json - containing offer ids and other attributes about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Requirements
  - pandas
  - numpy
  - sklearn
  - plotly
  - matplotlib
  
## Results

Goal of this project was to determine the following questions:

Q1. What features primarily influence customer use of the offer?

The feature importance assigned by three models indicates that the term of the membership is the most critical element influencing how customers respond to the offer. The top three variables for all three models were nearly identical for each offer type: 
- `tenure of the member`
- `income`
- `age`. 
However, the order of income and age changed based on the type of offer. 

Q2. Given the data available, including the offer attributes and user demographics, indicate whether a user would accept an offer?

I've decided to use a single model with offer type as a categorical variable to predict whether or not the consumer will respond to the offer, with 'offer success' as a target variable. This model achieved an `AUC score of around 78 percent`. 

I also created three separate models for each offer type, with the models for the bogo and discount offer types performing well. Despite the fact that the model performance of the informational offer was significantly worse, it is still acceptable in this stage of research.
