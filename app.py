import streamlit as st
import numpy as np
import pickle
symptom_mapping = {
    "acidity": 0,
    "indigestion": 1,
    "headache": 2,
    "blurred_and_distorted_vision": 3,
    "excessive_hunger": 4,
    "muscle_weakness": 5,
    "stiff_neck": 6,
    "swelling_joints": 7,
    "movement_stiffness": 8,
    "depression": 9,
    "irritability": 10,
    "visual_disturbances": 11,
    "painful_walking": 12,
    "abdominal_pain": 13,
    "nausea": 14,
    "vomiting": 15,
    "blood_in_mucus": 16,
    "Fatigue": 17,
    "Fever": 18,
    "Dehydration": 19,
    "loss_of_appetite": 20,
    "cramping": 21,
    "blood_in_stool": 22,
    "gnawing": 23,
    "upper_abdomain_pain": 24,
    "fullness_feeling": 25,
    "hiccups": 26,
    "abdominal_bloating": 27,
    "heartburn": 28,
    "belching": 29,
    "burning_ache": 30,
}


# Loading Diease Dectection Pickle File
f = open("DecisionTree-Model.pkl", "rb")
model_N = pickle.load(f)

# loading Medicine Recommendation Pickle file
f2 = open("drugTree.pkl", "rb")
model_med = pickle.load(f2)


# Defining a fucntion to convert user inputs and predict
def serviceValidation(selected_symptoms):
    # Convert the selected symptoms to a 30-element list of 1s and 0s
    inputs = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    for symptom in selected_symptoms:
        if symptom:
            inputs[symptom_mapping[symptom]] = 1

    # convert list to NumPy array
    inputs = np.array(inputs)
    inputs = inputs.reshape(1, -1)

    # Pass the inputs to your machine learning model and retrieve the predicted result
    predicted_result = model_N.predict(inputs)
    print(predicted_result[0])

    # Return the predicted result to the user
    return predicted_result[0].lower()


def Convert(selectedOptions):
    d = {"diarrhea": 1, "gastritis": 2, "arthritis": 3, "migraine": 4}
    g = {"MALE": 1, "FEMALE": 2}
    s = {"LOW": 1, "NORMAL": 2, "HIGH": 3}
    result = [
        d[selectedOptions[0]],
        selectedOptions[1],
        g[selectedOptions[2]],
        s[selectedOptions[3]],
    ]
    return result


def medicineValidation(selectedOptions):
    """Defining a function to recommend medicine"""
    inputs = np.array(selectedOptions)  # convert list to NumPy array
    inputs = inputs.reshape(1, -1)
    # Pass the inputs to your machine learning model and retrieve the predicted result
    recommend_Med = model_med.predict(inputs)
    # Return the predicted result to the user
    return recommend_Med[0]


import requests


def more_info(a):
    url = f"http://api.serpstack.com/search?access_key=ad95da0ffbf61a651aaf37540c20c5e4&query={a}"
    data = requests.get(url).json()
    return data["request"]["search_url"]

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://2.bp.blogspot.com/-GCSj6LSo4hg/XL7idXULnUI/AAAAAAAAACQ/YYAr9Tby0ikN6fERQqrjPnVkliSVASiYQCLcBGAs/s1600/ayurveda-background-3.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
def main():
    
    add_bg_from_url()
    
    st.markdown('<h1 style="color:black;">AYURVEDIC AGENCY</h1>', unsafe_allow_html=True)

    Gender = st.radio(":black[SELECT YOUR GENDER:]", [":black[MALE]", ":black[FEMALE]"])

    Age = st.number_input("",placeholder="Enter Your Age",min_value=0,max_value=100)

    Symptom1 = st.selectbox(
        "",
        [
            "acidity",
            "indigestion",
            "headache",
            "blurred_and_distorted_vision",
            "excessive_hunger",
            "muscle_weakness",
            "stiff_neck",
            "swelling_joints",
            "movement_stiffness",
            "depression,irritability",
            "visual_disturbances",
            "painful_walking",
            "abdominal_pain",
            "nausea",
            "vomiting",
            "blood_in_mucus",
            "Fatigue",
            "Fever",
            "Dehydration",
            "loss_of_appetite",
            "cramping,blood_in_stool",
            "gnawing,upper_abdomain_pain",
            "fullness_feeling",
            "hiccups",
            "abdominal_bloating",
            "heartburn",
            "belching",
            "burning_ache",
        ],index=None,placeholder="Select First Symptom",
    )

    Symptom2 = st.selectbox(
        "",
        [
            "acidity",
            "indigestion",
            "headache",
            "blurred_and_distorted_vision",
            "excessive_hunger",
            "muscle_weakness",
            "stiff_neck",
            "swelling_joints",
            "movement_stiffness",
            "depression,irritability",
            "visual_disturbances",
            "painful_walking",
            "abdominal_pain",
            "nausea",
            "vomiting",
            "blood_in_mucus",
            "Fatigue",
            "Fever",
            "Dehydration",
            "loss_of_appetite",
            "cramping,blood_in_stool",
            "gnawing,upper_abdomain_pain",
            "fullness_feeling",
            "hiccups",
            "abdominal_bloating",
            "heartburn",
            "belching",
            "burning_ache",
        ],index=None,placeholder="Select Second Symptom",
    )

    Symptom3 = st.selectbox(
        "",
        [
            "acidity",
            "indigestion",
            "headache",
            "blurred_and_distorted_vision",
            "excessive_hunger",
            "muscle_weakness",
            "stiff_neck",
            "swelling_joints",
            "movement_stiffness",
            "depression,irritability",
            "visual_disturbances",
            "painful_walking",
            "abdominal_pain",
            "nausea",
            "vomiting",
            "blood_in_mucus",
            "Fatigue",
            "Fever",
            "Dehydration",
            "loss_of_appetite",
            "cramping,blood_in_stool",
            "gnawing,upper_abdomain_pain",
            "fullness_feeling",
            "hiccups",
            "abdominal_bloating",
            "heartburn",
            "belching",
            "burning_ache",
        ],index=None,placeholder="Select Third Symptom",
    )

    Symptom4 = st.selectbox(
        "",
        [
            "acidity",
            "indigestion",
            "headache",
            "blurred_and_distorted_vision",
            "excessive_hunger",
            "muscle_weakness",
            "stiff_neck",
            "swelling_joints",
            "movement_stiffness",
            "depression,irritability",
            "visual_disturbances",
            "painful_walking",
            "abdominal_pain",
            "nausea",
            "vomiting",
            "blood_in_mucus",
            "Fatigue",
            "Fever",
            "Dehydration",
            "loss_of_appetite",
            "cramping,blood_in_stool",
            "gnawing,upper_abdomain_pain",
            "fullness_feeling",
            "hiccups",
            "abdominal_bloating",
            "heartburn",
            "belching",
            "burning_ache",
        ],index=None,placeholder="Select Fourth Symptom",
    )

    Severity = st.radio(":black[SELECT SEVERITY OF THE SYMPTOMS:]", [":green[LOW]", ":blue[NORMAL]", ":red[HIGH]"])

    if st.button("Submit"):
        with st.container(border=True):
            selected_symptoms = [Symptom1, Symptom2, Symptom3, Symptom4]
            diesease = serviceValidation(selected_symptoms).upper()
            st.subheader(':green[DETECTED DIESEASE IS : ]'+f':red[{diesease}]')
            d_info = more_info(diesease.lower())
            st.markdown("[More information]({})".format(d_info))

            selectedOptions = [diesease.lower(), Age, Gender, Severity]
            selectedOptions = Convert(selectedOptions)
            medicine = medicineValidation(selectedOptions).upper()

            st.subheader(':blue[RECOMMEDED MEDICINE FOR THE DIESEASE IS: ]'+f':red[{medicine}]')
            m_info = more_info(medicine.lower())
            st.markdown("[More information]({})".format(m_info))


if __name__ == "__main__":
    main()
