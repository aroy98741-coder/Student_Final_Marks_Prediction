# Student Final Marks Prediction

Predicts a student's **Final Exam Marks (out of 100)** from attendance, two internal tests,
assignment score and daily study hours. Includes EDA, four regression models, evaluation
plots, saved models, a prediction script, a Streamlit web app, tests and a notebook.

Dataset: `data/Final_Marks_Data.csv` (2,000 rows; 1,997 after removing 3 duplicate rows once Student_ID is dropped; downloaded from Kaggle).
Kaggle link: https://www.kaggle.com/datasets/sonalshinde123/student-academic-performance-dataset

## Project structure
```
Student_Performance_Prediction/
|-- data/Final_Marks_Data.csv        dataset
|-- src/
|   |-- config.py                    paths, column names, constants
|   |-- preprocessing.py             load/clean data, pipeline, train/test split
|   |-- eda.py                       exploratory data analysis -> results/figures
|   |-- train_models.py              trains + compares models -> models/ and results/
|   `-- predict.py                   predict marks for one student
|-- notebooks/student_performance_prediction.ipynb
|-- models/                          saved models (.pkl) - created by train_models.py
|-- results/                         figures/, model_results.csv, correlation_matrix.csv, ...
|-- tests/test_pipeline.py
|-- app.py                           Streamlit web app
|-- requirements.txt
`-- README.md
```

## Quick start (VS Code)
1. Install Python (python.org) and the VS Code **Python** extension.
2. Unzip, then File -> Open Folder -> `Student_Performance_Prediction`.
3. Open a terminal (Terminal -> New Terminal) and install libraries (one time):
   ```
   pip install -r requirements.txt
   ```
   (If `pip` is not found: `python -m pip install -r requirements.txt`; Mac/Linux: `python3`.)
4. Run the steps in order:
   ```
   python src/eda.py            # EDA: prints statistics, saves plots to results/figures
   python src/train_models.py   # trains 4 models, saves models/ and results/
   python src/predict.py        # example prediction
   streamlit run app.py         # web app (opens in your browser)
   pytest                       # run tests
   ```
   Custom prediction:
   ```
   python src/predict.py --attendance 90 --test1 30 --test2 35 --assignment 8 --hours 3
   ```
5. Notebook: open `notebooks/student_performance_prediction.ipynb` in VS Code, select a Python
   kernel, and click **Run All**.

> Run `train_models.py` before `predict.py` or the app, because they load `models/best_model.pkl`.

## Expected results
| Model | R2 (test) | MAE (marks) | CV R2 |
|---|---|---|---|
| Ridge / Linear Regression | 0.839 | 3.48 | 0.834 |
| Gradient Boosting | 0.821 | 3.65 | 0.815 |
| Random Forest | 0.796 | 3.94 | 0.797 |

Most important features (permutation importance): Internal Test 2, Internal Test 1, Attendance.
Ridge and Linear Regression give identical scores here, since the data is simple and
well-behaved. The linear models beating the tree models suggests the relationship is mostly linear.

## What each step does (for your presentation)
1. **preprocessing.py** - drops Student_ID, renames columns, removes duplicates, fills missing values
   (median) and scales features inside a Pipeline, so no information leaks from the test set.
2. **eda.py** - shape, types, missing values, duplicates, descriptive statistics, outliers (IQR),
   distributions, correlation heatmap, feature-vs-target scatter plots.
3. **train_models.py** - 80/20 split, 5-fold cross-validation, metrics (R2, MAE, RMSE), saves every
   model, plots actual-vs-predicted, residuals and feature importance.
4. **predict.py / app.py** - load the best model and predict marks for a new student.

## Common problems
- `No module named ...` -> run step 3 again in the same terminal.
- `best_model.pkl not found` -> run `python src/train_models.py` first.
- File not found -> the terminal must be inside the project folder.
- No plot windows appear -> normal; plots are saved as images in `results/figures/`.

## Limitations
- Predictions can be off by about 3-4 marks on average, more for individual students.
- Data may be synthetic - check the dataset description on Kaggle and say so if it is.
