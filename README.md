# 🎓 Student Placement Predictor

An AI-powered web app that predicts whether a student will get placed,  
built using **Logistic Regression** trained on **9,000 student records**.

## 🔗 Live Demo
👉 [Click here to try the app](YOUR_STREAMLIT_LINK_HERE)  
*(Replace this link after deploying on Streamlit Cloud)*

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.78% |
| ROC-AUC | 0.972 |
| Precision (Placed) | 95% |
| Recall (Placed) | 97% |

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Library:** scikit-learn
- **Web App:** Streamlit
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application (live prediction + charts) |
| `logistic_regression_placement.py` | Full model training code with detailed explanation |
| `requirements.txt` | Python dependencies |
| `student_placement_salary_elite_v2.csv` | Dataset — 9,000 student records, 20 features |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-placement-predictor.git
cd student-placement-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 🔍 Features Used for Prediction

- CGPA, Branch, College Tier
- Python, DSA, ML, Web Dev skills
- Coding, Communication, Aptitude scores
- Internships, Projects, Backlogs
- Resume Score, Skill Score

---

## 💡 How It Works

1. **Data Preprocessing** — Label encoding for categorical features, StandardScaler for normalization
2. **Model Training** — Logistic Regression on 80% of data (7,200 students)
3. **Evaluation** — Tested on 20% unseen data (1,800 students)
4. **Prediction** — Enter student details → get placement probability instantly

---

## 👨‍💻 Author
Built as a Machine Learning portfolio project.
