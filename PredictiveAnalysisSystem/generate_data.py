import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# 1. student_id: 1001 to 3000
student_id = np.arange(1001, 1001 + n)

# 2. age: 16 to 25 (skewed towards 18-22 typical college/high school)
age = np.random.choice(
    np.arange(16, 26),
    size=n,
    p=[0.05, 0.10, 0.20, 0.25, 0.18, 0.10, 0.05, 0.03, 0.02, 0.02]
)

# 3. gender: Male / Female
gender = np.random.choice(['Male', 'Female'], size=n, p=[0.51, 0.49])

# Latent diligence / conscientiousness factor to create realistic mild cross-correlations
diligence = np.random.normal(0, 1, size=n)

# 4. family_income: monthly in INR (e.g. 15,000 to 350,000, lognormal distribution)
# Median around 50k - 60k
log_income = np.random.normal(10.9, 0.65, size=n)
family_income = np.exp(log_income)
family_income = np.clip(np.round(family_income / 1000) * 1000, 15000, 350000).astype(int)

# 5. internet_access: Boolean, higher chance with higher income
prob_internet = 0.55 + 0.38 / (1 + np.exp(-(family_income - 40000) / 25000))
internet_access = np.random.binomial(1, np.clip(prob_internet, 0.5, 0.98), size=n).astype(bool)

# 6. attendance_percentage: 40 to 100
base_att = 78 + 8 * diligence + np.random.normal(0, 7.5, size=n)
attendance_percentage = np.clip(np.round(base_att, 1), 40.0, 100.0)

# 7. study_hours_per_day: 0 to 12
base_study = 3.8 + 1.4 * diligence + (attendance_percentage - 75) * 0.03 + np.random.normal(0, 1.3, size=n)
study_hours_per_day = np.clip(np.round(base_study, 1), 0.0, 12.0)

# 8. assignment_completion: 0 to 100
base_assign = 50 + 20 * (diligence + 1) / 2 + 0.25 * attendance_percentage + 1.5 * study_hours_per_day + np.random.normal(0, 6, size=n)
assignment_completion = np.clip(np.round(base_assign, 1), 0.0, 100.0)

# 9. previous_exam_score: 0 to 100
base_prev = 35 + 0.3 * attendance_percentage + 2.5 * study_hours_per_day + 0.2 * assignment_completion + 4 * diligence + np.random.normal(0, 7.5, size=n)
previous_exam_score = np.clip(np.round(base_prev, 1), 0.0, 100.0)

# 10. sleep_hours: 3 to 12 (students who study excessively late might get slightly less sleep)
base_sleep = 7.4 - 0.15 * (study_hours_per_day - 4) + np.random.normal(0, 1.1, size=n)
sleep_hours = np.clip(np.round(base_sleep, 1), 3.0, 12.0)

# 11. extracurricular_hours: 0 to 6
base_extra = 1.8 - 0.1 * study_hours_per_day + np.random.exponential(0.9, size=n)
extracurricular_hours = np.clip(np.round(base_extra, 1), 0.0, 8.0)

# 12. final_exam_score: target variable (0 to 100)
# Highly realistic combination:
# - Strong linear & non-linear baseline from previous performance and current preparation
# - Sleep sweet spot: optimal around 7-8 hours, penalty for <5 or >10 hours
# - Extracurricular sweet spot: moderate participation (1-2.5h) slightly beneficial / balanced, excessive (>4h) cuts into focus
# - Diminishing returns on extreme study hours (log/sqrt effect)
# - Internet access small positive utility (information access)
# - Realistic Gaussian noise

study_effect = 3.6 * np.sqrt(study_hours_per_day) + 0.8 * study_hours_per_day
prev_effect = 0.42 * previous_exam_score
att_effect = 0.15 * attendance_percentage
assign_effect = 0.12 * assignment_completion

# Non-linear sleep effect (inverted U-shape peaking around 7.5 hrs)
sleep_effect = -1.1 * ((sleep_hours - 7.5) ** 2) + 2.0

# Non-linear extracurricular effect (mild peak around 1.5 hrs, penalty if > 3.5 hrs)
extra_effect = -0.7 * ((extracurricular_hours - 1.5) ** 2) + 1.2

internet_effect = np.where(internet_access, 1.8, -0.8)

latent_score = (
    8.5 +
    prev_effect +
    study_effect +
    att_effect +
    assign_effect +
    sleep_effect +
    extra_effect +
    internet_effect +
    np.random.normal(0, 4.8, size=n) # natural exam-day variability / noise
)

final_exam_score = np.clip(np.round(latent_score, 1), 0.0, 100.0)

df = pd.DataFrame({
    'student_id': student_id,
    'age': age,
    'gender': gender,
    'attendance_percentage': attendance_percentage,
    'study_hours_per_day': study_hours_per_day,
    'previous_exam_score': previous_exam_score,
    'assignment_completion': assignment_completion,
    'sleep_hours': sleep_hours,
    'extracurricular_hours': extracurricular_hours,
    'internet_access': internet_access,
    'family_income': family_income,
    'final_exam_score': final_exam_score
})

# Save to CSV in workspace
csv_path = 'student_academic_performance.csv'
df.to_csv(csv_path, index=False)
print("Saved CSV successfully. Total rows:", len(df))
print("\n--- FIRST 20 RECORDS ---")
print(df.head(20).to_string())

print("\n--- SUMMARY STATISTICS ---")
print(df.describe().to_string())

print("\n--- CORRELATION MATRIX ---")
numeric_cols = df.select_dtypes(include=[np.number, bool]).columns
corr = df[numeric_cols].corr()
print(corr['final_exam_score'].sort_values(ascending=False).to_string())
print("\nFull correlation matrix:\n", corr.round(3).to_string())
