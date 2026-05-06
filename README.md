# 🎓 EduSense – Student Performance Predictor

EduSense is a Machine Learning-powered web application built using Streamlit that predicts a student's final exam marks based on academic performance indicators such as attendance, internal test scores, assignment marks, and study hours.

The application provides:
- 📊 Final score prediction
- 🏆 Grade estimation
- 📈 Performance insights
- 💡 Personalized improvement tips
- 🔬 ML model comparison (Random Forest & Linear Regression)
- 🎨 Modern premium UI using Streamlit custom styling

---

## 🚀 Features

- Predict student final exam marks instantly
- Interactive and modern UI
- Random Forest & Linear Regression models
- Real-time score preview
- Personalized smart recommendations
- Data visualization and analytics
- Fully deployable on Streamlit Cloud

---

## 🧠 Machine Learning Models Used

- Random Forest Regressor
- Linear Regression

The models are trained using:
- Attendance Percentage
- Internal Test Scores
- Assignment Score
- Daily Study Hours

---

## 📂 Project Structure

```bash
Student-Performance-Predictor/
│
├── student_performance_app.py
├── marks_analysis.ipynb
├── Final_Marks_Data.csv
├── student_performance_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
```

### 2️⃣ Create virtual environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate environment:

### Windows
```bash
venv\Scripts\activate
```

### Mac/Linux
```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run student_performance_app.py
```

---

## 🌐 Live Demo

🚀 Access the live dashboard here:

👉 [Student Performance Predictor Dashboard](https://edusenseperformancepredictor.streamlit.app)


---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Attendance (%) | Student attendance percentage |
| Internal Test 1 | Marks out of 40 |
| Internal Test 2 | Marks out of 40 |
| Assignment Score | Marks out of 10 |
| Daily Study Hours | Hours studied daily |
| Final Exam Marks | Target variable |

---

## 📷 Screenshots

Add screenshots of:
- Home Page
- Prediction Result
- Nerd Zone Analytics
- Visualizations

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📌 Future Improvements

- User authentication
- Database integration
- Deep Learning models
- PDF report generation
- Student performance tracking dashboard
- Multi-user support

---

## 👨‍💻 Author

**Shobhit Sharma**

Aspiring Data Analyst & Machine Learning Enthusiast

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub!