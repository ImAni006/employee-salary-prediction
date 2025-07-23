

```markdown
# 🧠 Employee Salary Prediction Model

This project is a **machine learning pipeline** that predicts whether an individual's income exceeds $50K/year based on demographic and work-related features. It includes **data preprocessing**, multiple **classification models**, and a **Streamlit web app** for deployment.

---

## 📂 Project Structure

```

📁 employee-salary-prediction/
├── data/
│   └── adult.csv
├── salary\_model.pkl
├── streamlit\_app.py
├── preprocessing.py
├── model\_training.ipynb
└── README.md

````

---

## 📊 Problem Statement

- Predict whether a person earns more than $50K per year.
- Based on features like age, education, occupation, work hours, etc.
- Built using various supervised ML algorithms and deployed as a Streamlit web app.

---

## ✅ Preprocessing Steps

- Handled missing values (`?` replaced with labels like `Others`, `not_listed`)
- Removed rare or irrelevant categories (e.g., `'Armed-Forces'`, `'Without-pay'`)
- Dropped low-information features and filtered records with low `educational-num`
- Outliers removed using the **IQR method**
- Applied **Label Encoding** and **One-Hot Encoding**
- Scaled numerical features using **StandardScaler**

---

## 🤖 Machine Learning Algorithms Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- All models evaluated using **accuracy score** and **visualized with graphs**

---

## 🚀 Streamlit Web App

- Built a user-friendly web interface using **Streamlit**
- Takes user input for features like age, education, hours-per-week, etc.
- Applies the same preprocessing as during training
- Predicts income category (`>50K` or `<=50K`)
- Run with:
  ```bash
  streamlit run streamlit_app.py
````

---

## 📚 Requirements

Install the following Python packages before running:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

---

## 🧪 How to Run

1. Clone this repo or download the files.
2. Train the model using `model_training.ipynb` *(or use provided `salary_model.pkl`)*.
3. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```
4. Interact with the app through your browser.

---

## 💡 Challenges Faced

* Cleaning a noisy dataset with unclear missing values
* Encoding mixed-type categorical data correctly
* Managing preprocessing consistency during deployment
* Integrating model input/output logic into a web interface

---

## 📚 References

* UCI Adult Dataset: [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)
* Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
* Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

---

## ✨ Author

**Aniruddha Pareek**
*Machine Learning Enthusiast*

