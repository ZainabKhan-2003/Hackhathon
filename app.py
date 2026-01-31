# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load the saved model and encoder
# -------------------------------
model = joblib.load("logistic_model.pkl")
ohe = joblib.load("onehot_encoder.pkl")

# -------------------------------
# App title and description
# -------------------------------
st.title("Student Dropout Early Warning System")
st.write("""
This app predicts whether a student is at risk of dropping out 
based on their academic and personal data.
""")

# -------------------------------
# Input fields
# -------------------------------
# Categorical inputs
gender = st.selectbox("Gender", ["M", "F"])
NationalITy = st.selectbox("Nationality", ["KW", "Jordan", "SaudiArabia"])
PlaceofBirth = st.selectbox("Place of Birth", ["KW", "Jordan", "SaudiArabia"])
StageID = st.selectbox("Stage ID", ["lowerlevel", "MiddleSchool", "HighSchool"])
GradeID = st.selectbox("Grade ID", ["G-02","G-03","G-04","G-05","G-06","G-07","G-08","G-09"])
SectionID = st.selectbox("Section ID", ["A","B","C"])
Topic = st.selectbox("Topic", ["IT","Math","Science","English","Arabic","Geology","Biology","Chemistry","French"])
Semester = st.selectbox("Semester", ["F","S"])
Relation = st.selectbox("Parent Relation", ["Father","Mum"])
ParentAnsweringSurvey = st.selectbox("Parent Answering Survey", ["Yes","No"])
ParentschoolSatisfaction = st.selectbox("Parent School Satisfaction", ["Good","Bad"])
StudentAbsenceDays = st.selectbox("Student Absence Days", ["Under-7","Above-7"])

# Numeric inputs
raisedhands = st.number_input("Raised Hands", min_value=0, max_value=100, value=10)
VisITedResources = st.number_input("Visited Resources", min_value=0, max_value=100, value=10)
AnnouncementsView = st.number_input("Announcements View", min_value=0, max_value=100, value=5)
Discussion = st.number_input("Discussion", min_value=0, max_value=100, value=5)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict"):
    # Create dataframe for the model
    user_input = pd.DataFrame({
        "gender": [gender],
        "NationalITy": [NationalITy],
        "PlaceofBirth": [PlaceofBirth],
        "StageID": [StageID],
        "GradeID": [GradeID],
        "SectionID": [SectionID],
        "Topic": [Topic],
        "Semester": [Semester],
        "Relation": [Relation],
        "raisedhands": [raisedhands],
        "VisITedResources": [VisITedResources],
        "AnnouncementsView": [AnnouncementsView],
        "Discussion": [Discussion],
        "ParentAnsweringSurvey": [ParentAnsweringSurvey],
        "ParentschoolSatisfaction": [ParentschoolSatisfaction],
        "StudentAbsenceDays": [StudentAbsenceDays]
    })

    # Split columns for encoder
    cat_cols = ["gender","NationalITy","PlaceofBirth","StageID","GradeID",
                "SectionID","Topic","Semester","Relation","ParentAnsweringSurvey",
                "ParentschoolSatisfaction","StudentAbsenceDays"]
    num_cols = ["raisedhands","VisITedResources","AnnouncementsView","Discussion"]

    # Apply OneHotEncoder to categorical columns
    user_cat_ohe = ohe.transform(user_input[cat_cols])
    user_cat_df = pd.DataFrame(user_cat_ohe, columns=ohe.get_feature_names_out(cat_cols))

    # Combine numeric and encoded categorical columns
    user_final = pd.concat([user_input[num_cols].reset_index(drop=True), user_cat_df.reset_index(drop=True)], axis=1)

    # Make prediction
    prediction = model.predict(user_final)

    # Display result
    if prediction[0] == "1":
        st.error("⚠️ This student is at risk of dropping out!")
    else:
        st.success("✅ This student is safe!")
