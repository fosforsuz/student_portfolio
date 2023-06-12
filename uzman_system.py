from bardapi import Bard
import pandas as pd
import pickle
from prompt_toolkit import HTML
from sklearn.preprocessing import OneHotEncoder
from googletrans import Translator



"""
Visit https://bard.google.com/
F12 for console
Session: Application → Cookies → Copy the value of __Secure-1PSID cookie and assign to token.
"""
token = ""
bard = Bard(token=token)

academic_success = pickle.load(open("academic_success.sav", "rb"))
family_relation = pickle.load(open("family_relation.sav", "rb"))
life_balance = pickle.load(open("life_balance.sav", "rb"))

# read student data
data = pd.read_csv("test.csv")

# Create titles from excel file structure
title = [
    "dalc",
    "G1",
    "G2",
    "pstatus",
    "walc",
    "absences",
    "activities",
    "failures",
    "famsize",
    "freetime",
    "goout",
    "higher",
    "internet",
    "schoolsup",
    "studytime",
    "health",
    "family_relation",
]


def create_uzman_system(value):
    pdf_text = ""
    
    academic_value = data[
        [
            "freetime",
            "failures",
            "studytime",
            "absences",
            "internet",
            "higher",
            "schoolsup",
            "dalc",
            "walc",
            "G1",
            "G2",
        ]
    ]
    ohe = OneHotEncoder(sparse_output=False)
    # Fit and transform data_x
    encoder = ohe.fit_transform(academic_value.iloc[:, 4:7])
    # Concat the encoder and data_x from 4 to 7 with iloc
    academic_value = pd.concat(
        [
            academic_value.iloc[:, :4],
            pd.DataFrame(encoder),
            academic_value.iloc[:, 7::],
        ],
        axis=1,
    )

    family_value = value[
        [
            "freetime",
            "goout",
            "dalc",
            "walc",
            "health",
            "activities",
            "famsize",
            "pstatus",
        ]
    ]
    ohe = OneHotEncoder(sparse_output=False)
    encoded_data = ohe.fit_transform(family_value.iloc[:, 5::])
    family_value = pd.concat(
        [family_value.iloc[:, :5], pd.DataFrame(encoded_data)], axis=1
    )

    life_value = value[
        [
            "freetime",
            "goout",
            "dalc",
            "walc",
            "famrel",
            "activities",
            "famsize",
            "pstatus",
        ]
    ]
    ohe = OneHotEncoder(sparse_output=False)
    encoded_data = ohe.fit_transform(life_value.iloc[:, 5::])
    life_value = pd.concat([life_value.iloc[:, :5], pd.DataFrame(encoded_data)], axis=1)

    for x in range(len(value.values)):
        # predict academic success, family relation and life balance
        academic_predict = academic_success.predict(
            academic_value.values[x].reshape(1, -1)
        )
        family_predict = family_relation.predict(family_value.values[x].reshape(1, -1))
        life_predict = life_balance.predict(life_value.values[x].reshape(1, -1))
        # create text for profile
        text = create_text_for_profile(
            value.iloc[x], academic_predict, family_predict, life_predict
        )

        # generate completion
        completion = generate_completaion(text)
        
        with open("student_portfolio.md", "a") as file:
            file.write(("## Student " + str(x + 1) + "\n"))
            file.write(f"{completion}\n")
            file.write("**********************************************************************\n")
        
    print(pdf_text)


def create_text_for_profile(data, academic_predict, family_predict, life_predict):
    return (
        f"Student portfolio:\n"
        + f"\tAcademic achievement: {academic_predict} (out of 20) \n"
        + f"\tFamily relationships: {family_predict}\n"
        + f"\tLife balance: {life_predict}\n"
        + f"\tStudent information:\n"
        + f"\t\t{data['freetime']} free time\n"
        + f"\t\t{data['goout']} going out\n"
        + f"\t\t{data['dalc']} weekday alcohol consumption\n"
        + f"\t\t{data['walc']} weekend alcohol consumption\n"
        + f"\t\t{data['health']} good health\n"
        + f"\t\t{data['activities']} participating in activities\n"
        + f"\t\t{data['famsize']} family size\n"
        + f"\t\t{data['pstatus']} parents' cohabitation status\n"
    )


def generate_completaion(text):
    return bard.get_answer(text)['content']

if __name__ == "__main__":
    create_uzman_system(data)
