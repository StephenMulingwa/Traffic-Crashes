# Predict The Primary Contributory Cause of Car Accidents

## Business Understanding

This project aims to support the Vehicle Safety Board by leveraging data from the Chicago Police Department's E-Crash system to identify patterns and primary contributory causes of traffic accidents. Through predictive modeling and data analysis, the project seeks to generate actionable insights to enable data-driven interventions to enhance road safety, optimize resource allocation, and inform policy decisions.

The transportation sector plays a pivotal role in any country's social and economic growth. However, the increasing number of vehicles on the roads has heightened the challenges of traffic management and safety. By efficiently analyzing traffic crash data, this project will contribute to significant advancements in road safety measures, public awareness, strategic policy-making, and effective resource deployment, ultimately fostering safer streets and proactive accident prevention.

##### Problem Statement

Traffic accidents in Chicago remain a significant public health and economic challenge, despite advancements in vehicle safety technology, traffic management systems, and the availability of data from the Chicago Police Department’s E-Crash system. The lack of detailed analysis and predictive capabilities hinders effective mitigation efforts and limits understanding of the primary contributory causes. This project aims to develop a predictive model to identify key factors driving traffic crashes, uncover actionable insights, and enable data-driven strategies to enhance road safety. By supporting the Vehicle Safety Board, the project seeks to reduce accidents, optimize resource allocation, and make Chicago’s roads safer for everyone.

##### Objectives

1. Predict the primary contributory cause of a car accident.
2. Identify high-risk locations for traffic accidents to prioritize interventions like increased patrols, signage, or infrastructure improvements.
3. Analyze the role of driver behaviors, such as speeding or distracted driving, to guide public awareness campaigns and enforcement strategies
4. Understand how the timing of accidents impacts crash severity to optimize resource deployment during high-risk periods.

##### Research Questions

1. What are the most common primary contributory causes of traffic accidents?
2. Which locations in Chicago experience the highest frequency of traffic accidents, and what patterns contribute to their high risk?
3. How do driver behaviors, such as speeding or distracted driving, influence the likelihood and severity of traffic accidents?
4. How does the timing of accidents (e.g., time of day, day of the week, or season) affect crash severity, and what insights can be drawn to optimize resource deployment?

##### Stakeholders

- Government and Public Agencies i.e Vehicle Safety Board
- Policymakers and Urban Planners
- General Public and Road Users
- Public Health and Safety Organizations

## Data Understanding

This dataset was sourced from kaggle and it has 794956 and 48 columns.The dataset has been used by other data scientists before and is accessible for anyone to look through it, it is updated often. The data is in csv format inside a folder named data. I then read through the data using the panda's library in order to get a data frame as our output . The dataset has data recorded in different data type including float, intergers and objects.

## Data Analysis

We have conducted an Exploratory Data Analysis (EDA) to better understand the patterns, trends, and impacts of traffic accidents on both the economy and public health. Our analysis focused on various key aspects of accidents, including their distribution across injury types, economic costs, contributing causes, high-risk locations, and temporal patterns.

The visualizations and analysis performed here provide a comprehensive understanding of the factors influencing traffic accidents. By examining the relationship between accidents and various elements such as injury severity, damage costs, contributing causes, locations, speed, time, and more, we are better equipped to develop strategies to mitigate accidents, enhance public safety, and address the economic impacts of traffic incidents.
Here are some of the visualization.https://public.tableau.com/app/profile/boniface.ngechu/viz/Book1_17379105122570/Dashboard1?publish=yes 
## Data Visualization
### 1.  Crash Distribution by Day of Week.
![image](https://github.com/user-attachments/assets/7c36981a-9911-4619-b595-86032cb8186d)

### 2. Accident Occurrence each Month.
![image](https://github.com/user-attachments/assets/005506a7-2914-42ab-98b7-1f79c31b68e7)





## Model Performance Overview
Through predictive modeling and data analysis, the project seeks to generate actionable insights to enable data-driven interventions to enhance road safety, optimize resource allocation, and inform policy decisions.

#### ** 1.Recurrent Neural Network (RNN):**
1. **Performance Metrics:**
   - **Accuracy:** 51.1%  
   - **Precision:** 66.2%  
   - **Recall:** 31.2%  
   - **F1-Score:** 0.425  
   - **Cohen's Kappa:** 0.331 (moderate agreement)  
   - **MCC:** 0.350 (moderate correlation)

2. **Strengths:**
   - **Sequential Data Handling:** RNN, with its LSTM layers, effectively captured temporal dependencies and patterns in the data.
   - **Balanced Generalization:** Training and validation curves suggest no signs of overfitting, indicating that the model generalizes well.
   - **Moderate Agreement:** Cohen's Kappa and MCC show that the RNN performs significantly better than random guessing.

3. **Weaknesses:**
   - **Low Recall:** The model struggles to capture all true positive cases, which could lead to missed critical insights in business decisions.
   - **F1-Score:** While better than random guessing, the balance between precision and recall still needs improvement for real-world deployment.

4. **Implication:**  
   The RNN's performance demonstrates moderate predictive power, making it useful for identifying key patterns in sequential data, especially where false positives are more tolerable than false negatives.

---

#### ** 2.Convolutional Neural Network (CNN):**
1. **Performance Metrics:**
   - **Accuracy:** 48.4%  
   - **Precision:** 64.9%  
   - **Recall:** 28.0%  
   - **F1-Score:** 0.372  
   - **Cohen's Kappa:** 0.268 (fair agreement)  
   - **MCC:** 0.289 (weak correlation)

2. **Strengths:**
   - **Spatial Feature Extraction:** CNN effectively recognized local patterns in structured data through convolutional layers.
   - **Precision:** The model minimized false positives better than the RNN, which is advantageous in scenarios where incorrect classifications are costly.

3. **Weaknesses:**
   - **Recall:** The CNN performed poorly in capturing true positive cases, indicating difficulty in identifying less frequent classes.
   - **Overfitting:** Evidence of overfitting after 0.5 epochs suggests the model struggles to generalize to unseen data.
   - **Lower Agreement:** Cohen's Kappa and MCC scores indicate weaker reliability compared to the RNN.

4. **Implication:**  
   The CNN model is less suited for this problem due to its limited ability to generalize and its struggles with recall. Its use may result in overlooking critical insights, which could lead to suboptimal business outcomes.

### Objectives
#### Predict the primary contributory cause of a car accident.
 -The primary objective of this project is to predict the primary contributory cause of car accidents, providing the Vehicle Safety Board with actionable insights to enhance road safety, optimize resource allocation, and inform evidence-based policy decisions

### Conclusion

The Recurrent Neural Network (RNN) outperforms the Convolutional Neural Network (CNN) in predicting the primary contributory cause of accidents. The RNN achieves higher accuracy (0.51) and a better Cohen's Kappa score (0.33) compared to the CNN’s accuracy (0.48) and Cohen’s Kappa score (0.27). These results indicate that the RNN model is more reliable and aligns better with the true labels in the data. While both models show moderate performance, the RNN demonstrates a stronger ability to generalize the relationships in the dataset.

### Recommendations

-Adopt the RNN Model: Since the RNN performs better, it should be the model of choice for predicting accident causes in this application.

-Optimize the RNN Model: Consider fine-tuning the hyperparameters or experimenting with more advanced architectures like LSTMs or GRUs to improve performance further.

-Address Class Imbalance: The dataset's imbalance may affect model performance. Techniques such as weighted loss functions or resampling methods should be explored.

-Enhance Features: Investigate the inclusion of additional features or the refinement of existing ones to capture more relevant patterns in the data.

-Evaluate Other Models: For robustness, additional models (e.g., Gradient Boosting or Transformer-based architectures) should be explored and compared.

-Monitor Overfitting: Pay close attention to training dynamics to ensure models generalize effectively, as evidenced by CNN overfitting in earlier epochs.
