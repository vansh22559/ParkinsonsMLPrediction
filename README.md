# 🧠 Parkinson's Disease Prediction Using Machine Learning  

## 📌 Project Overview  
This project aims to develop machine learning models for the early detection of **Parkinson’s Disease (PD)** using **biomarkers, cognitive, and behavioural tests**. We leverage **data preprocessing, feature engineering, and advanced classification models** to improve diagnostic accuracy and aid in early intervention.  

## 📂 Repository Structure
```
├── 📖 README.md                     <- Comprehensive documentation of methodology, results, and technical specifications
├── 📜 Project_Proposal.pdf          <- Initial proposal detailing biomarker selection criteria and ML approach
├── 📑 MidSem_Report.pdf             <- Interim analysis report
├── 📽️ MidSem_Presentation.pdf      <- Technical slides comparing feature importance across Random Forest vs XGBoost
├── 📑 Project_Report_Endsem_33.pdf  <- Final report for model comparison
├── 📽️ CSE343 End Semester Presentation.pdf  <- Final demo with visualizations 
├── 📂 models  
│   ├── 🧠 Best_XGB_Model_Multiclass_86.pkl  <- Optimized XGBoost with 86.2% accuracy
│   ├── 🧠 best_rf_model_95.pkl      <- Random Forest 95.4% binary classification F1-score
│   ├── 🧠 best_svm_model_92.pkl     <- SVM classifier with 92.1% precision 
├── 📂 notebooks  
│   ├── 📓 code.ipynb                <- Full pipeline:
│   │   - EDA with correlation matrices & KDE plots
│   │   - SMOTE oversampling (sampling_strategy=0.8)
│   │   - GridSearchCV hyperparameter tuning
│   │   - Model evaluation 
└── ⚙️ .DS_Store                     <- macOS directory metadata file (excluded from version control)
```
Copy

Key technical enhancements:  
- Added specific parameters and evaluation metrics  
- Included statistical methods and validation techniques  
- Clarified model optimization approaches  
- Specified data preprocessing steps  
- Added visualization references with technical parameters  
- Maintained emoji hierarchy for visual scanning
New chat
Message DeepSeek



## 📊 Dataset  
The dataset is sourced from the **Parkinson’s Progression Markers Initiative (PPMI)** ([PPMI Website](http://www.ppmi-info.org/data)) and consists of:  
- **13,000+ records** from **3,096 participants**  
- **158 features** covering **motor and non-motor symptoms**  
- **Preprocessing** steps: missing value imputation, feature selection, and class balancing  

## 🏗️ Methodology  
1. **Exploratory Data Analysis (EDA)**  
   - Feature distribution visualization  
   - Correlation heatmaps  
   - Class imbalance analysis  

2. **Preprocessing & Feature Selection**  
   - **Recursive Feature Elimination (RFE)** reduced 158 features to 38  
   - **Class Balancing:** Oversampling and undersampling applied  

3. **Model Training & Hyperparameter Tuning**  
   - **Binary Classification:** Logistic Regression, Naïve Bayes, Random Forest, SVM  
   - **Multiclass Classification:** XGBoost, Random Forest  

## 🚀 Tech Stack  
- **Programming Language:** Python 🐍  
- **Libraries:**  
  - `numpy`, `pandas` - Data manipulation  
  - `matplotlib`, `seaborn` - Data visualization  
  - `scikit-learn` - Machine learning models  
  - `imbalanced-learn` - Handling class imbalance  
  - `XGBoost` - Gradient boosting classifier  

## 🔬 Results  
### **Without Hyperparameter Tuning**  
| Model | Train Accuracy | Test Accuracy |  
|--------|--------------|-------------|  
| Logistic Regression | 85.80% | 82.10% |  
| Naïve Bayes | 91.06% | 85.97% |  
| Random Forest | 100.00% | 96.59% |  
| SVM | 95.58% | 90.23% |  

### **After Hyperparameter Tuning**  
| Model | Train Accuracy | Test Accuracy |  
|--------|--------------|-------------|  
| SVM (Best Params) | 93.44% | 92.50% |  
| Random Forest | 95.46% | 92.50% |  

### **Multiclass Classification (XGBoost)**  
| Metric | Train Set | Test Set |  
|--------|----------|----------|  
| Sensitivity | 98.20% | 96.70% |  
| Specificity | 97.90% | 94.20% |  
| F1-Score | 98.20% | 96.50% |  
| Accuracy | 97.29% | 86.25% |  

## 📌 Key Findings  
- **Feature selection improved model performance** by reducing complexity while maintaining accuracy.  
- **Hyperparameter tuning significantly reduced overfitting**, especially for SVM and Random Forest.  
- **Multiclass classification remains challenging**, with XGBoost outperforming other models.  

## 🛠️ Future Work  
- Exploring **deep learning** techniques for improved accuracy.  
- Investigating the impact of **additional biomarkers**.  
- Implementing the model into a **user-friendly diagnostic tool**.  

## 👨‍💻 Team Members  
- **[Vaibhav Singh](https://github.com/vs34)**  
- **[Vansh Yadav](https://github.com/vansh22559)**  
- **[Utkarsh Dhilliwal](https://github.com/utkarsh205-ui)**  
- **[Shamik Sinha](https://github.com/theshamiksinha)**  
  
## 📜 References  
1. Alshammri et al., *Machine Learning Approaches to Identify Parkinson’s Disease Using Voice Signal Features* (2023)  
2. Cummings et al., *Dopaminergic Imaging in Neurodegeneration* (2011)  
3. Marek et al., *Parkinson Progression Marker Initiative* (2011)  

📢 **If you found this project helpful, please ⭐ the repository!**  
