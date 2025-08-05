# 🏠 Predicting High Booking Rate Airbnb Listings with Ensemble Machine Learning

This project aims to predict which Airbnb listings are likely to have high booking rates using advanced machine learning techniques. The solution leverages the Inside Airbnb dataset and applies extensive data preprocessing, feature engineering, and ensemble modeling strategies to deliver a robust predictive pipeline.

> 🎯 Our team (Group 15) secured **3rd place out of 20+ teams** in a university-wide competition with a final **ROC AUC of 0.91176**.

---

## 📌 Problem Statement

Classify Airbnb listings as **high booking rate** or not, based on listing-level features such as amenities, pricing, host responsiveness, and location. This supports hosts and stakeholders in optimizing listings to maximize occupancy.

---

## 🧠 Approach

### 🔹 1. Data Source
- **Dataset**: Inside Airbnb (2023 snapshot) – ~92,000 listings
- **Target Variable**: `high_booking_rate` (binary classification)

### 🔹 2. Data Preprocessing
- Imputation for missing values (mean/mode)
- Log-transformations for skewed features (e.g., price)
- Category consolidation (e.g., rare market/property types)

### 🔹 3. Feature Engineering
- `TF-IDF` vectorization on house rules & amenities
- `Sentiment Analysis` of host description using **TextBlob**
- Normalized price-per-capita features
- Availability ratios & calendar-based features
- Binary flags for key amenities (WiFi, AC, kitchen, etc.)
- KMeans clustering on geolocation

### 🔹 4. Modeling & Evaluation

#### ✅ Models Explored
- Logistic Regression  
- Decision Trees  
- Random Forest  
- K-Nearest Neighbors  
- **XGBoost**  
- **CatBoost**  
- **Stacking Ensemble** (XGBoost + CatBoost with Logistic Regression meta-learner)

#### 🧪 Imbalance Handling
- Stratified train-test split  
- **SMOTE** oversampling  

#### ⚙️ Hyperparameter Tuning
- `Optuna` for XGBoost  
- `GridSearchCV` for CatBoost  

#### 📊 Evaluation Metrics
- **ROC AUC: 0.91176**  
- Accuracy: ~87%  
- Precision, Recall, F1-score  
- 5-fold Stratified Cross-Validation  
- Confusion Matrix  

#### 🔍 Model Interpretability
- **SHAP** (Shapley Additive Explanations)  
- Key drivers: review scores, host response rate, pricing, amenities, host tenure

---

## 🏆 Competition Result

- 🥉 **Placed 3rd out of 20+ teams**
- Top 6 teams' AUC scores were extremely close (range: ~0.916 to ~0.908)
- Validated with leaderboard screenshot (Group 15)

---

## 🔍 Key Insights

- Responsive hosts and moderately priced listings performed best
- Strict cancellation policies surprisingly correlated with higher bookings
- Review quality and host experience were strong predictors

---

## 📁 File Structure
├── final_submission.ipynb         # Final model & preprocessing pipeline
├── EDA.ipynb                      # Exploratory Data Analysis
├── high_booking_rate_group15.csv # Final model submission
├── group15_report.docx           # Official report (submission)
├── annotated-Group15_*.pdf       # Instructor-reviewed report

---

## 🛠️ Technologies Used

- Python 3.9+  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- **xgboost**, **catboost**  
- **optuna**, **imbalanced-learn (SMOTE)**  
- **shap**  
- **textblob**, **nltk**  

---

## 🚀 How to Run

1. Clone this repository  
2. Install dependencies with:  
   ```bash
   pip install -r requirements.txt

👨‍💻 Contributors
	•	Chanamallu Venkata Chandrasekhar Vinay
	•	Team 15 | BUDT758T – Data Mining & Predictive Analytics | Spring 2025

⸻

📜 License

This project is for educational purposes only under the University of Maryland’s BUDT758T course. Not intended for commercial use.

