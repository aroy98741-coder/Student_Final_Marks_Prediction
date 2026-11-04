"""Central settings: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Final_Marks_Data.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
for _d in (FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Original (long) column names in the CSV -> short names used in the code
RENAME = {
    "Attendance (%)": "attendance",
    "Internal Test 1 (out of 40)": "internal_test_1",
    "Internal Test 2 (out of 40)": "internal_test_2",
    "Assignment Score (out of 10)": "assignment_score",
    "Daily Study Hours": "study_hours",
    "Final Exam Marks (out of 100)": "final_marks",
}
ID_COLUMNS = ["Student_ID"]
TARGET = "final_marks"
FEATURES = ["attendance", "internal_test_1", "internal_test_2",
            "assignment_score", "study_hours"]
