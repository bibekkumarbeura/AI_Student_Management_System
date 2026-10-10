import os
import pandas as pd
import streamlit as st

from model import predict_performance, get_performance_category


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Student Management System",
    page_icon="\U0001f393",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "students.csv")

COLUMNS = [
    "Roll_No",
    "Name",
    "Branch",
    "Semester",
    "Attendance",
    "Study_Hours",
    "Previous_Marks"
]


# --------------------------------------------------
# Initialize CSV File
# --------------------------------------------------

def initialize_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_FILE, index=False)


# --------------------------------------------------
# Load Student Data
# --------------------------------------------------

def load_students():
    initialize_csv()

    try:
        students = pd.read_csv(CSV_FILE)

        for column in COLUMNS:
            if column not in students.columns:
                students[column] = None

        return students[COLUMNS]

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)


# --------------------------------------------------
# Save Student Data
# --------------------------------------------------

def save_students(students):
    students.to_csv(CSV_FILE, index=False)


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("\U0001f393 AI Student Management System")

st.write(
    "Manage student records and predict academic performance "
    "using Machine Learning."
)

st.divider()


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select a Page",
    [
        "Dashboard",
        "Add Student",
        "View Students",
        "Predict Performance"
    ]
)


# Load student data
students = load_students()


# --------------------------------------------------
# 1. Dashboard
# --------------------------------------------------

if menu == "Dashboard":

    st.header("\U0001f4ca Dashboard")

    total_students = len(students)

    if not students.empty:

        attendance = pd.to_numeric(
            students["Attendance"],
            errors="coerce"
        )

        marks = pd.to_numeric(
            students["Previous_Marks"],
            errors="coerce"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Students",
            total_students
        )

        col2.metric(
            "Average Attendance",
            (
                f"{attendance.mean():.1f}%"
                if attendance.notna().any()
                else "N/A"
            )
        )

        col3.metric(
            "Average Marks",
            (
                f"{marks.mean():.1f}"
                if marks.notna().any()
                else "N/A"
            )
        )

        st.subheader("Student Records")

        st.dataframe(
            students,
            width="stretch"
        )

        chart = students[
            ["Name", "Previous_Marks"]
        ].copy()

        chart["Previous_Marks"] = pd.to_numeric(
            chart["Previous_Marks"],
            errors="coerce"
        )

        chart = chart.dropna()

        if not chart.empty:

            chart = chart.set_index("Name")

            st.subheader("\U0001f4c8 Previous Marks Chart")

            st.bar_chart(chart)

    else:

        st.info(
            "No student records available. "
            "Please add a student first."
        )


# --------------------------------------------------
# 2. Add Student
# --------------------------------------------------

elif menu == "Add Student":

    st.header("\u2795 Add New Student")

    with st.form("add_student_form"):

        roll_no = st.number_input(
            "Roll Number",
            min_value=1,
            step=1
        )

        name = st.text_input("Student Name")

        branch = st.selectbox(
            "Branch",
            ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil", "Other"]
        )

        semester = st.selectbox(
            "Semester",
            [1, 2, 3, 4, 5, 6, 7, 8]
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

        previous_marks = st.number_input(
            "Previous Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        submitted = st.form_submit_button(
            "Add Student"
        )

    if submitted:

        if not name.strip():

            st.error("Please enter the student name.")

        else:

            students = load_students()

            if (
                pd.to_numeric(
                    students["Roll_No"],
                    errors="coerce"
                ) == roll_no
            ).any():

                st.error(
                    "This Roll Number already exists."
                )

            else:

                new_student = pd.DataFrame(
                    [
                        {
                            "Roll_No": int(roll_no),
                            "Name": name.strip(),
                            "Branch": branch,
                            "Semester": semester,
                            "Attendance": attendance,
                            "Study_Hours": study_hours,
                            "Previous_Marks": previous_marks
                        }
                    ]
                )

                students = pd.concat(
                    [students, new_student],
                    ignore_index=True
                )

                save_students(students)

                st.success(
                    f"Student {name} added successfully!"
                )

                st.rerun()


# --------------------------------------------------
# 3. View, Search and Delete Students
# --------------------------------------------------

elif menu == "View Students":

    st.header("\U0001f468\u200d\U0001f393 Student Records")

    students = load_students()

    if not students.empty:

        search_text = st.text_input(
            "\U0001f50d Search by Student Name or Roll Number"
        )

        if search_text.strip():

            search_value = search_text.strip().lower()

            filtered_students = students[
                students["Name"]
                .astype(str)
                .str.lower()
                .str.contains(search_value, na=False)
                |
                students["Roll_No"]
                .astype(str)
                .str.contains(search_value, na=False)
            ]

        else:

            filtered_students = students

        st.dataframe(
            filtered_students,
            width="stretch"
        )

        csv_data = filtered_students.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="\U0001f4e5 Download Student Records (CSV)",
            data=csv_data,
            file_name="students.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("\U0001f5d1\ufe0f Delete Student")

        roll_numbers = pd.to_numeric(
            students["Roll_No"],
            errors="coerce"
        ).dropna().astype(int).tolist()

        if roll_numbers:

            selected_roll = st.selectbox(
                "Select Roll Number",
                roll_numbers
            )

            if st.button("Delete Selected Student"):

                students = students[
                    pd.to_numeric(
                        students["Roll_No"],
                        errors="coerce"
                    ) != selected_roll
                ]

                save_students(students)

                st.success(
                    f"Student with Roll Number "
                    f"{selected_roll} deleted successfully."
                )

                st.rerun()

    else:

        st.info(
            "No students found. Please add a student first."
        )


# --------------------------------------------------
# 4. Predict Student Performance
# --------------------------------------------------

elif menu == "Predict Performance":

    st.header("\U0001f916 AI Student Performance Prediction")

    st.write(
        "Enter the student's details to estimate "
        "their academic performance."
    )

    with st.form("prediction_form"):

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

        previous_marks = st.number_input(
            "Previous Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        predict_button = st.form_submit_button(
            "Predict Performance"
        )

    if predict_button:

        try:

            predicted_marks = predict_performance(
                attendance,
                study_hours,
                previous_marks
            )

            category = get_performance_category(
                predicted_marks
            )

            st.success(
                "Performance prediction completed!"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Marks",
                f"{predicted_marks:.2f}%"
            )

            col2.metric(
                "Performance Category",
                category
            )

            st.progress(
                min(
                    max(int(predicted_marks), 0),
                    100
                )
            )

            st.info(
                "Note: This is an estimate from your trained "
                "machine-learning model, not a guaranteed result."
            )

        except FileNotFoundError:

            st.error(
                "Model file not found. "
                "Please run train_model.py first."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "AI Student Management System | Machine Learning Project"
)
