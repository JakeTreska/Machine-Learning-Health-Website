import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

all_data = pd.read_csv("healthcare-dataset-stroke-data.csv")


with open("voting_model_ML",'rb') as file:
    model = pickle.load(file)


def preprocessing(df):

    # change boolean to 0/1
    df["ever_married"] = df["ever_married"].map({"Not Married": 0, "Married": 1})
    df["Residence_type"] = df["Residence_type"].map({"Urban": 0, "Rural": 1})
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["hypertension"] = df["hypertension"].map({"False": 0, "True": 1})
    df["heart_disease"] = df["heart_disease"].map({"False": 0, "True": 1})

    # finish one hot encoding for smoke status and work type
    smoke_dictionary = {"Unknown": 0, "Never": 1, "Formally smoked": 2, "Smokes": 3}
    smoke_location = smoke_dictionary[smoker]
    df[f'smoking_status_{smoke_location}'] = 1.0
    work_type_dictionary = {"Government Job": 'work_type_Govt_job', "Private": 'work_type_Private',
                            "Self-Employed": 'work_type_Self-employed', "Child": 'work_type_children'}
    work_location = work_type_dictionary[work_type]
    df[work_location] = 1.0


    # log transformation
    df['avg_glucose_level'] = np.log(df['avg_glucose_level'])
    df['bmi'] = np.log(df['bmi'])
    df['age/bmi'] = df['age'] / df['bmi']




# WEBSITE CODE

st.set_page_config(page_title="Stroke Risk Predictor", page_icon=":tada:")

# ---- HEADER SECTION ----
with st.container():
    temp_bmi = 0
    temp_glucose = 0

    # st.subheader("WADDUP FUCKERS :wave:")
    st.title("Stroke Risk Predictor")
    st.subheader(
        "It never hurts to check..."
    )


# ---- About us

st.header('', divider='red')

st.markdown("""
Knowledge is power when it comes to your health. We believe that understanding your individual risk factors for stroke is the first step toward a healthier future. Our mission is to empower you with knowledge and awareness, enabling you to take proactive steps to reduce your risk of stroke.

Stroke is a serious medical condition, but many of its risk factors are manageable through lifestyle changes and early intervention. By providing us with some basic personal information, you can gain valuable insights into your unique risk profile. Our user-friendly tool will analyze your data and provide you with personalized recommendations to reduce your risk of stroke.

Remember, this tool is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a healthcare professional for a comprehensive assessment. However, our Stroke Risk Checker can serve as a useful starting point to help you take control of your health and well-being.

Your health matters, and taking proactive steps today can make a significant difference in your future. Let's work together to reduce the risk of stroke and promote a healthier, happier life. Get started now and take the first step towards a stroke-free tomorrow.





Personal data will NOT be collected.

Heart failure detection coming soon...
"""
            )

with st.container():
    st.header('', divider='red')
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('_Step 1:_')
        st.write("##")
        st.write(
            """
            Enter relevant information for each of the data required.
            """
        )
    with right_column:
        st.subheader('_Step 2:_')
        st.write("##")
        st.write(
            """
            Press Enter and recieve your results.
            """
        )
    st.header('', divider='red')

# THIS IS FOR AGE
st.title("Age")

age = st.slider("Age", min_value=1, max_value=100, value=25)

st.write(f"Age: {age}")

# THIS IS FOR GENDER
st.title("Gender")

gender = st.selectbox("Gender", ["--Unselected--", "Male", "Female"])

st.write(f"Gender: {gender}")

# THIS IS FOR MARRIAGE
st.title("Married")

marriage = st.radio("Marital Status", ["Married", "Not Married"], index = None)

st.write(f"Marriage Status: {marriage}")

# THIS IS FOR housing
st.title("Residence")

residence = st.selectbox("Residence", ["--Unselected--", "Urban", "Rural"])

st.write(f"Residence: {residence}")

# This is for Avg Glucose

st.title("Average Glucose Level")
st.info("If you don't know your Glucose value, average values range from 70-140 mg/dL")

average_glucose = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)

st.write(f"Average Glucose Level: {average_glucose}")

# THIS IS FOR HYPERTENSION

st.title("Hypertension")

hypertension = st.radio("Hypertension", ["True", "False"], index = None)

st.write(f"Hypertension Status: {hypertension}")

# THIS IS FOR BMI
st.title("BMI Value")
bmi = st.number_input("BMI", min_value=0.0, step=0.1)

st.write(f"Entered BMI: {bmi}")

# THIS IS FOR HEART DISEAS
st.title("Heart Disease")

heart_disease = st.radio("Heart Disease", ["True", "False"],  index = None)

st.write(f"Heart Disease Status: {heart_disease}")

# THIS IS FOR WORK TYPE
st.title("Work Type")
work_type = st.radio("Work Type",['Government Job','Private','Self-Employed','Child'])
st.write(f"Work Type Status: {work_type}")


# THIS IS FOR SMOKE
st.title("Smoker")
smoker = st.selectbox("Smoker?", ["--Unselected--", "Unknown", "Never", "Formally smoked", "Smokes"])

st.write(f"Smoking status: {smoker}")

# ENTER BUTTON

if (
        age != None and
        gender != "--Unselected--" and
        marriage != None and
        residence != "--Unselected--" and
        average_glucose is not None and
        hypertension is not None and
        bmi is not None and
        heart_disease is not None and
        work_type is not None and
        smoker != "--Unselected--"
):
    enter_button = st.button("Enter")
    if enter_button:
        # Process the entered data here

        temp_bmi = bmi
        temp_glucose = average_glucose

        data = {"gender": gender, "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
                "ever_married": marriage, "Residence_type": residence,
                "avg_glucose_level": average_glucose,
                "bmi": bmi, 'age/bmi': 0,
                "smoking_status_0": 0.0, "smoking_status_1": 0.0,
                "smoking_status_2": 0.0, "smoking_status_3": 0.0,
                "work_type_Govt_job": 0.0,
                "work_type_Private": 0.0,
                "work_type_Self-employed": 0.0,
                "work_type_children": 0.0}

        df = pd.DataFrame([data])
        preprocessing(df)

        st.header('', divider='grey')

        prediction = model.predict(df)
        if (prediction == 1):
            st.header(':red[Your prediction is positive for a stroke.]')
        else:

            st.header(f":green[Your prediction is negative for a stroke!]")


        st.header('', divider='grey')

        fig1 = plt.figure(figsize=(16, 4))
        sns.pointplot(data = df,x="age", y=temp_bmi, color = 'green')
        sns.lineplot(data=all_data, x="age", y="bmi", hue="stroke")
        fig2 = plt.figure(figsize=(16, 4))
        sns.pointplot(data=df, x="age", y=temp_glucose, color='green')
        sns.lineplot(data=all_data, x="age", y="avg_glucose_level", hue="stroke")
        fig3 = plt.figure(figsize=(16, 4))
        sns.barplot(x=all_data.stroke, y=all_data.hypertension)

        st.write("Your data in the graphs below is marked by the green dot")
        st.pyplot(fig1)
        st.write(
            "This graph displays the trend between BMI (Body Mass Index) and age in relation to the probability of experiencing a stroke. As the data illustrates, there's a certain correlation between higher BMI values and an increased risk of stroke, especially as one advances in age. However, it's essential to note that while BMI can be an indicative factor, individual risks may vary based on genetics, lifestyle, and other health conditions.")
        st.pyplot(fig2)
        st.write(
            "The graph showcases the relationship between average glucose levels, age, and the likelihood of a stroke. Elevated glucose levels, often indicative of conditions like diabetes, can heighten the risk of vascular complications, including strokes. As seen in the trend, older individuals with higher glucose levels might be at a heightened risk. Still, individual circumstances and overall health play crucial roles in determining the actual risk.")

        st.pyplot(fig3)
        st.write(
            "This bar graph presents the association between hypertension (high blood pressure) and the incidence of stroke. Hypertension is a well-known risk factor for cardiovascular diseases, including stroke. As the graph reveals, individuals with more severe hypertension levels have a progressively increased likelihood of experiencing a stroke. Regular blood pressure checks and management are vital for mitigating this risk.")

else:
    st.warning("Please fill in all the required fields to enable the 'Enter' button.")



