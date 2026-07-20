# 🏦 Loan Approval Prediction System

A Machine Learning-powered web application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant details. The application is built using **Flask**, **Scikit-learn**, and **Pandas**, with a clean web interface for real-time predictions.

🌐 **Live Demo:** https://loanapprovalprediction-0yma.onrender.com

---

## 📌 Features

- ✅ Predicts Loan Approval (Approved / Rejected)
- ✅ Real-time prediction using a trained Machine Learning model
- ✅ User-friendly web interface
- ✅ Displays prediction confidence
- ✅ Explains possible reasons for approval/rejection
- ✅ Responsive UI
- ✅ Flask backend
- ✅ Deployed on Render

---

## 🚀 Live Demo

👉 https://loanapprovalprediction-0yma.onrender.com

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

### Backend
- Flask
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Deployment
- Render
- Gunicorn

---

## 📂 Project Structure

```
LoanApprovalPrediction/
│
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── runtime.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── screenshots/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/VijayKumar-Kareti/LoanApprovalPrediction.git
```

### Go to Project Folder

```bash
cd LoanApprovalPrediction
```

### Create Virtual Environment

#### Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 📊 Machine Learning Workflow

- Data Collection
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Serialization using Joblib
- Deployment with Flask

---

## 🧠 Input Features

- Age
- Gender
- Marital Status
- Education
- Employment Type
- Monthly Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Dependents
- Self Employed

---

## 📈 Prediction Output

The system predicts:

- ✅ Loan Approved
- ❌ Loan Rejected

It also provides:

- Prediction Probability
- Confidence Score
- Reason for the Prediction

---

## 📸 Screenshots

Add screenshots of your application inside the `screenshots` folder and update this section.

Example:

```
screenshots/
│── home.png
│── prediction.png
│── result.png
```

---

## 🚀 Deployment

The application is deployed using **Render**.

Live URL:

https://loanapprovalprediction-0yma.onrender.com

---

## 🔮 Future Enhancements

- User Authentication
- Prediction History
- Loan Eligibility Dashboard
- Explainable AI (SHAP/LIME)
- Email Notification
- Multiple ML Model Comparison
- PDF Report Generation
- REST API Support

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 👨‍💻 Author

**Venkata Vijay Kumar Kareti**

GitHub:
https://github.com/VijayKumar-Kareti

LinkedIn:
(Add your LinkedIn Profile)

---

## ⭐ Support

If you like this project,

⭐ Star this repository

and share it with others!

---

## 📜 License

This project is developed for educational and learning purposes.
