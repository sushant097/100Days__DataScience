# 100 Days of Data Science


<p align="center">
  <img src="images/cover_photo.png" alt="Image 1" width="500" height="500">
  <!-- <img src="https://github.com/user-attachments/assets/2a2302a3-eff7-4461-aa76-c77e5e8541a0" alt="Image 2" width="300" height="300"> -->
</p>



Welcome to my **100 Days of Data Science** journey! 🚀

Over the following days, I'll be diving deep into data science, working on a variety of datasets, and exploring different topics within the field. Each time, I’ll be tackling new challenges, from data exploration and preprocessing to building and fine-tuning machine learning models. My goal is to document this journey and share my learnings, code, and insights with the community.

## What's Inside

- **Daily Projects**: Each time, you'll find a new project where I've applied data science techniques to solve a specific problem.
- **Code and Notebooks**: All my code and Jupyter notebooks will be available here, so you can follow along or use them as a reference.
- **Datasets**: Links to the datasets I’m using, along with a brief explanation of the problem statement and objectives.
- **Learning Resources**: I’ll also be sharing articles, tutorials, and resources that I found helpful along the way.


## How to Use This Repository

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/sushant097/100Days__DataScience.git
   ```
2. **Navigate Through the Folders**: Each day’s project will have its own folder with detailed explanations and code.
3. **Run the Notebooks**: If you want to try out the code yourself, simply open the notebooks and run them in your preferred environment. Also kaggle notebook link is provided in some cases. 


## Day 1: Women's E-Commerce Clothing Reviews

On the first day, I worked with the **Women's E-Commerce Clothing Reviews** dataset. I explored the data, handled preprocessing tasks, and built a machine learning model to predict product recommendations. You can find all the details and code for this project in the Day 1 folder.

* Github Implementation NotebooK: [Github Notebook](Implementation/Day1/transforming-review-data-into-features.ipynb)
* Kaggle Notebook: [Kaggle Notebook](https://www.kaggle.com/code/sushant097/transforming-review-data-into-features/)
* Dataset Link: [Dataset link](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)


## Day 2: Predict Adult Income

On Day 2, I worked with the Adult income dataset. I focused on exploring the data, performing preprocessing, and building a machine-learning model to predict adult income >50k or not.

Mainly worked with two models, logistic Regression and random Forest Classifier, are compared. The data seems imbalanced, so the random forest classifier improves the model's accuracy a bit. 

Learned and implemented :
* Data Encoding and feature scaling improves model performance. 
* Increase model complexity if the data is complex and accuracy is not improved with a simpler model. 
* Hyperparameter tuning methods like grid search improve the model accuracy as it finds the best parameters that work for the model for the given dataset.
* Visualized the feature importance of the model: Model interpretation.
* Search and listed other ways to improve model accuracy like SMOTE for data imbalance case, model class weight adjustment such that model can focus on minority class, Ensembling methods (like stacking, boosting) is another option.

* Github Implementation NotebooK: [Github Notebook](Implementation/Day1/transforming-review-data-into-features.ipynb)
* Kaggle Notebook: [Kaggle Notebook](https://www.kaggle.com/code/sushant097/day2-100daysofdatascience/)
* Dataset Link: [Dataset link](http://archive.ics.uci.edu/dataset/2/adult)

![](images/Day1_result.png)


## Day3: Online Retail Customer Segmentation


🔍 **Explored Online Retail Data for Customer Segmentation Using Machine Learning**

- **Dataset Used**: Online Retail II (2009-2010), containing transactional data including Invoice, StockCode, CustomerID, Quantity, and Price.
  
- **Data Preparation**:
  - Loaded the dataset and performed initial exploration to understand its structure.
  - Cleaned the data by removing duplicates and handling missing values.
  - Conducted feature engineering to create relevant metrics like `TotalSpent`, `NumOrders`, and `TotalQuantity` for each customer.

- **Exploratory Data Analysis (EDA)**:
  - Visualized the distribution of key features like `TotalSpent` and explored relationships using scatter plots and pairplots.
  - Utilized Seaborn to create attractive and insightful visualizations.

- **Data Normalization**:
  - Applied standardization to ensure all features contributed equally in the clustering process.

- **Customer Segmentation with K-Means Clustering**:
  - Determined the optimal number of clusters using the Elbow Method.
  - Applied K-Means clustering to segment customers based on their purchasing behavior.
  - Visualized the customer segments and analyzed their characteristics.

- **Key Insights**:
  - Successfully segmented customers into distinct groups, each with unique purchasing patterns.
  - Identified potential target groups for personalized marketing strategies.


* Github Implementation NotebooK: [Github Notebook](Implementation/Day3/day3-online-retail-prediction.ipynb)
* Kaggle Notebook: [Kaggle Notebook](https://www.kaggle.com/code/sushant097/day3-online-retail-customer-segmentation/)
* Dataset Link: [Dataset link](http://archive.ics.uci.edu/dataset/352/online+retail)

![](images/Day3_result.png)


## Day4: Bank marketing dataset to predict customer subscription behavior

🚀 On Day 4 I focused on a bank marketing dataset to predict customer subscription behavior. Here's what I accomplished:

🔍 Learned and implemented:

- 🧹 **Data Preprocessing:** Applied label encoding to handle categorical variables and standard scaling to normalize feature ranges.

- 📊 **Exploratory Data Analysis (EDA):** Conducted detailed univariate and bivariate analysis to uncover patterns and relationships in the data.

- 🤖 **Logistic Regression:** Built and interpreted a binary classification model to predict customer subscriptions, analyzing feature importance based on model coefficients.

- 🌳 **Random Forest Classifier:** Compared Logistic Regression with a Random Forest model, which provided better performance in identifying potential subscribers.

- 📈 **Feature Importance Visualization:** Visualized how different features contribute to the prediction, helping to understand the model's decision-making process.

💡 Takeaway:

- Combining simple models like Logistic Regression with more complex ones like Random Forest can offer valuable insights and improved accuracy in predictive tasks.

- Advanced data visualization techniques like univariate, bivariate, box plot, and pair plot help in understanding different attributes of the dataset, contributing to building a more robust machine learning model.

**Resources:**

* **[Notebook](./Implementation/Day%204/day4-predict-deposit-of-bank.ipynb)**

* **[Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full/data)**

<p align="center">
  <img src="images/Day_4_result.png" alt="Image 1" width="600" height="400">
  <img src="images/Day_4_correlation_heatmap.png" alt="Image 2" width="300" height="300">
</p>


## Day 5 of 100 Days of Data Science Challenge

### 🚀 Dataset: Dry Bean Dataset

Today, I focused on classifying different types of beans based on geometric and shape-related features using machine learning. Below are the tasks I completed:

### 🔍 Steps Implemented:

- 🧹 **Data Preprocessing**: Removed redundant features by analyzing high correlations to prevent multicollinearity and improve model efficiency.
  
- 📊 **Exploratory Data Analysis (EDA)**: Conducted feature exploration using pair plots and a correlation heatmap to discover patterns and feature relationships.

- 🌳 **Random Forest Classifier**: Built a Random Forest classification model to predict the bean classes with an accuracy of **92.6%**.

- 📈 **Feature Importance Analysis**: Visualized the feature importance, identifying the most relevant features contributing to the prediction.

- 📉 **Confusion Matrix**: Generated a confusion matrix to evaluate model performance by examining correct and incorrect predictions for each class.

### 💡 Takeaway:

Removing redundant features and performing feature importance analysis significantly improved the Random Forest model’s accuracy. Visualization tools like heatmaps and pair plots were essential for understanding data patterns and enhancing model interpretation.

### Some Results:
![Image 1](images/Day_5_result1.png)
![Image 2](images/Day5_result2.png)
![Image 2](images/Day5_result3.png)

**Resources:**
* **[Kaggle Notebook](https://www.kaggle.com/code/sushant097/day-5-dry-bean-dataset-analysis/)**
* **[Dataset](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset)**



## Day 6 of 100 Days of Data Science Challenge

### Tackling Overfitting in Machine Learning

![](images/Day6_overfittinginml.jpg)


**What is Overfitting?**  
Overfitting happens when a model performs exceptionally well on the training data but struggles with new, unseen data. Essentially, the model becomes too complex and starts capturing noise and outliers instead of the true underlying patterns. In other words, it "memorizes" the data rather than learning from it.

**How can we reduce Overfitting?**  
Here are a few techniques to prevent overfitting:
- **Regularization:** This adds a penalty to the complexity of the model. Two popular types are L1 (Lasso) and L2 (Ridge) regularization. They help by constraining or shrinking the coefficient values, which reduces the model's complexity.
- **Cross-validation:** Using techniques like k-fold cross-validation ensures that the model is evaluated on different portions of the data, making it more robust.
- **Simpler models:** Choosing simpler models with fewer parameters can reduce the chances of overfitting.
- **Early stopping:** This involves stopping the training process before the model starts overfitting the training data.
- **Data augmentation:** In cases like image classification, artificially increasing the size of the training data by applying transformations like rotation or flipping can help the model generalize better.

Learning to prevent overfitting is essential for building models that not only perform well on training data but also on new data. 

## Day 7 of 100 Days of Data Science Challenge

### Understanding the Bias-Variance Tradeoff in Machine Learning


The **bias-variance tradeoff** is a crucial concept in machine learning that helps us understand the performance of our models in terms of prediction error. This tradeoff occurs when balancing two sources of error:

- **Bias**: The error caused by a model that is too simple and cannot capture the underlying patterns in the data. High bias leads to **underfitting**, where the model performs poorly on both the training data and new, unseen data.

- **Variance**: The error caused by a model that is too complex and overly sensitive to the training data. High variance leads to **overfitting**, where the model fits even the noise in the data, resulting in poor performance on new data.

### Key Concept:
- **Underfitting**: When the model has high bias and cannot capture the relationships in the data.
- **Overfitting**: When the model has high variance and fits the training data too closely, including noise.

The goal in machine learning is to **minimize the total error** by finding a balance between bias and variance, i.e., creating a model that generalizes well to new, unseen data.

### Visualizing the Bias-Variance Tradeoff:

![](images/Day7_biasvariance.png)


In the above graph:
- As **model complexity** increases, **bias** decreases, but **variance** increases.
- The **total error** (black curve) is the sum of errors from bias and variance, and the optimal model complexity is where the total error is minimized.

### How to Handle the Bias-Variance Tradeoff:
- **Increase complexity** to reduce bias (e.g., using a more powerful model).
- **Reduce complexity** to decrease variance (e.g., using regularization techniques).
- Use techniques like **cross-validation** to ensure the model generalizes well.

This tradeoff is key to building a model that neither underfits nor overfits and performs well on both the training data and unseen data.


## Day 8 of 100 Days of Data Science: Handling Data Imbalance


When we're working with datasets, we'll often come across situations where one class has many more samples than the other(s). This is called **data imbalance**. For example, in a dataset of credit card transactions, there may be 95% legitimate transactions and only 5% fraudulent ones. A model trained on this data might simply predict "legitimate" every time and achieve high accuracy, but it would fail to catch fraud, which is the most important outcome.

Here’s how we can handle it:

### 1. **Resampling the Dataset**
   - **Oversampling**: Add more examples of the minority class by duplicating or creating synthetic examples (using techniques like SMOTE).
   - **Undersampling**: Reduce the majority class by removing some examples.

### 2. **Using Different Metrics**
   - Instead of using accuracy, we should look at metrics like:
     - **Precision**: How many of the predicted positives are correct.
     - **Recall**: How many actual positives were identified.
     - **F1-Score**: Harmonic mean of precision and recall.
     - **AUC-ROC**: Area under the curve that shows how well the model distinguishes between classes.

### 3. **Modifying Algorithms**
   - **Cost-sensitive learning**: Adjust the algorithm so that it penalizes the model more for getting the minority class wrong. This makes the model pay more attention to the minority class.

### 4. **Using Ensemble Methods**
   - Models like **Random Forest** or **Gradient Boosting** can combine multiple decision trees, making them more robust to imbalanced data.

---

Here’s a simple  code to demonstrate each technique:

![](images/Day8_code.png)

### Detailed Comments:

1. **Data generation**: We generate a synthetic dataset where 90% of the data belongs to the majority class, and only 10% belongs to the minority class.
2. **Train-test split**: We divide the dataset into training and test sets.
3. **Base model**: We train a RandomForest classifier on the imbalanced dataset to show how a model performs without handling imbalance.
4. **SMOTE**: We apply SMOTE, an oversampling technique, to balance the classes by generating synthetic data for the minority class.
5. **Undersampling**: We undersample the majority class by randomly selecting a subset of examples equal to the number of minority class examples. This ensures that both classes are equally represented.


## Day 9 of 100 Days of Data Science: 𝐀/𝐁 𝐓𝐞𝐬𝐭𝐢𝐧𝐠 – 𝐓𝐡𝐞 𝐊𝐞𝐲 𝐭𝐨 𝐃𝐚𝐭𝐚-𝐃𝐫𝐢𝐯𝐞𝐧 𝐃𝐞𝐜𝐢𝐬𝐢𝐨𝐧𝐬 🔍📊


### **What is A/B Testing?**

A/B testing is basically a way to compare two different versions of something—like a website, email, or product feature—to figure out which one works better. It’s like running a mini-experiment, where you show one version (Version A) to half of your audience and a second version (Version B) to the other half, then see which version gets better results.

### **An Everyday Example:**
Imagine you run an **online store**. You want to sell more items, so you're trying to figure out which “Buy Now” button works better. You have two different designs:
- **Version A**: A red button with the text “Buy Now.”
- **Version B**: A green button with the text “Shop Now.”

You don’t want to just guess which button will get more people to make a purchase, so you decide to test it out using A/B testing.

### **How to Run an A/B Test:**

1. **Split Your Audience**:  
   First, you divide your website visitors into two random groups. One group (Group 1) will see **Version A** with the red button, and the other group (Group 2) will see **Version B** with the green button. This way, each group gets a different experience, but everything else about the website stays the same.

2. **Track What Happens**:  
   Let the test run for some time, like a week. During this time, you track how many people from each group actually click the button and buy something.
   - **Group 1 (Red Button)**: Out of 1000 people, 100 make a purchase (which means a **10% conversion rate**).
   - **Group 2 (Green Button)**: Out of 1000 people, 120 make a purchase (which means a **12% conversion rate**).

   So it looks like the green button (Version B) is doing better since more people clicked and bought something. But we need to be sure this difference didn’t happen by chance.

3. **Is the Difference Real or Just Luck?**  
   To know for sure if the green button is better, we need to check if the results are **statistically significant**. This means we want to see if the difference in conversion rates is big enough to confidently say it’s because of the button and not just random luck. This is where we use something called a **p-value**. 

4. **Check for Statistical Significance**:  
   The **p-value** helps you figure out if the difference between the red and green button is real or just a coincidence. If the p-value is less than 0.05, it means there's only a 5% chance the difference happened randomly. In this case, if the p-value is below 0.05, we can confidently say that **Version B (green button)** really is better.

5. **Make Your Decision**:  
   If the test shows that the green button is truly better, you can now make it the default for everyone who visits your website. You know from the test that this change will likely lead to more purchases!

### **Breaking Down Some Key Terms:**

- **Conversion Rate**: The percentage of visitors who complete the action you want, like making a purchase. In our example, it was 10% for the red button and 12% for the green button.
- **Control Group (A)**: This is the group that sees the original version of the button (red in this case).
- **Test Group (B)**: This group sees the new version (green button).
- **Statistical Significance**: This is a way to measure if the difference between the two groups is real and not just due to chance.
- **p-value**: A number that tells you how likely it is that the difference happened by random chance. If it’s less than 0.05, you can be pretty confident that the difference is meaningful.

### **Why A/B Testing Matters:**

A/B testing helps you make decisions based on **data**, not just guesses or instincts. Instead of assuming what will work better, you test it with real users and see the results for yourself. This means your decisions are backed by facts, and you’re more likely to make improvements that actually matter.

### Code Example:
![](images/Day9_code.png)


## Day 10 of 100 Days of Data Science: Implement Linear Regression algorithm from scratch 


In linear regression, the goal is to model the relationship between the input features $(X)$ and the target variable $(y)$ by fitting a linear equation to the observed data. The model can be expressed as:

$$
y = X \theta
$$

Where:
- $(X)$ is the matrix of input features (including the bias term).
- $(\theta)$ is the vector of parameters (coefficients), which includes the bias term and the feature coefficients.
- $(y)$ is the vector of target values (the outputs we want to predict).

### Normal Equation

The Normal Equation provides a closed-form solution to compute the optimal parameters (\(\theta\)) that minimize the cost function (Mean Squared Error). The cost function is defined as:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

Where:
- $(m)$ is the number of training examples.
- $(h_{\theta}(x^{(i)}))$ is the predicted value for the $(i)$-th training example.
- $(y^{(i)})$ is the actual value for the $(i)$-th training example.

To minimize this cost function, we use the Normal Equation:

$$
\theta = (X^T X)^{-1} X^T y
$$

Where:
- $(X^T)$ is the transpose of the feature matrix $(X)$.
- $((X^T X)^{-1})$ is the inverse of the product of $(X^T)$ and $(X)$.
- $(y)$ is the target variable vector.


$$
RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}
$$

Where:
- $(y^{(i)})$ is the actual value for the $(i)$-th training example.
- $(\hat{y}^{(i)})$ is the predicted value for the $(i)$-th training example.
- $(m)$ is the number of data points.

The Root Mean Squared Error (RMSE) gives us an idea of how well the model fits the training data. A lower RMSE indicates a better fit of the model.


#### Gradient Descent (Alternative Approach):
While the Normal Equation provides a direct solution, you can also solve linear regression using **Gradient Descent**. However, this approach is iterative and requires careful tuning of hyperparameters such as the learning rate and number of iterations.

In Gradient Descent, the idea is to iteratively update the model parameters $(\theta)$ to minimize the cost function (Mean Squared Error in our case). The cost function is:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2
$$

Where:
- $(m)$ is the number of training examples.
- $h_{\theta}(x^{(i)})$ = $\theta_0$ + $\theta_1 x^{(i)}$ is the hypothesis (predicted value).

The gradient of the cost function with respect to $(\theta)$ is:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

This gradient tells us the direction and magnitude to adjust \(\theta\) to reduce the cost. We update \(\theta\) as:

$$
\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

Where $(\alpha)$ is the learning rate that controls how big the steps are during the update.


The code for this is:

![](images/Day10_code.png)


#### Explanation of Code:
1. **Normal Equation**:
   - We first compute $(\theta)$ using the closed-form solution with the Normal Equation.
   - The predicted values and RMSE are calculated and printed.

2. **Gradient Descent**:
   - A function `gradient_descent` is implemented. It iteratively updates $(\theta)$ using the gradient of the cost function.
   - After running for `n_iterations`, the final $(\theta)$ values are obtained.
   - Predictions are made using the learned $(\theta)$, and RMSE is computed and printed.

### Mathematical Explanation:

- **Cost Function**:  
  In linear regression, the cost function (Mean Squared Error) is minimized to find the best-fitting line. The cost function is:

$$
  J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2
$$

  Where:
  - $h_{\theta}(x^{(i)})$ = $theta_0$ + $theta_1 x^{(i)}$ is the prediction.
  - \(m\) is the number of training examples.

- **Gradient Descent Update Rule**:
  To minimize the cost function, we update \(\theta\) using the following rule:

$$
  \theta := \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

  This rule is applied repeatedly (for a fixed number of iterations) to get the optimal $(\theta)$.

#### Output:

The output will show the estimated coefficients and RMSE for both the **Normal Equation** and **Gradient Descent** methods.

For example:
```
Estimated coefficients using Normal Equation (theta): [[4.21509616]
                                                       [2.77011339]]
RMSE on the training set (Normal Equation): 0.8981005311027566

Estimated coefficients using Gradient Descent (theta): [[4.2075467 ]
                                                        [2.80339251]]
RMSE on the training set (Gradient Descent): 0.8983479400556326
```

#### Conclusion:

- **Normal Equation**: Provides an exact solution to linear regression by solving the equation $\theta$ = $(X^T X)^{-1} X^T y$. It works well for small to medium datasets but can be computationally expensive for very large datasets.
  
- **Gradient Descent**: Iteratively finds the solution by updating $(\theta)$ in the direction of the negative gradient. It's more scalable for large datasets but requires careful tuning of the learning rate and number of iterations.

Both methods yield similar results, with slight differences due to the iterative nature of gradient descent.