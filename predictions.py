## apis for prediction model

import sqlite3
import numpy 
import pickle




def preprocess_input(features):
    (
        age,
        blood_pressure,
        cholesterol,
        max_heart_rate,
        oldpeak,
        gender,
        chest_pain_type,
        fasting_blood_sugar,
        resting_ecg,
        exercise_angina,
        st_slope
    )= features



    # Initialize final input dict with all columns set to 0
    
    input_dict = {
        "age": float(age),
        "trestbps": float(blood_pressure),
        "chol": float(cholesterol),
        "thalch": int(max_heart_rate),
        "oldpeak": float(oldpeak),
        "sex_Male": 0,
        "cp_atypical angina": 0,
        "cp_non-anginal": 0,
        "cp_typical angina": 0,
        "fbs_True": 0,
        "restecg_normal": 0,
        "restecg_st-t abnormality": 0,
        "exang_True": 0,
        "slope_flat": 0,
        "slope_upsloping": 0
    }
    # initialization for categorical values
    #gender
    if(gender=='Male'):
        input_dict["sex_Male"]=1
    else:
        input_dict["sex_Male"]=0   


    # chest pain type
    if(chest_pain_type == 'Angina' or chest_pain_type=='angina'):
        input_dict["cp_atypical angina"]=0
        input_dict["cp_non-anginal"]=0
        input_dict["cp_typical angina"]=1
    elif(chest_pain_type == 'Atypical angina' or chest_pain_type=='atypical angina'):
        input_dict["cp_atypical angina"] = 1
        input_dict["cp_non-anginal"]= 0
        input_dict["cp_typical angina"]= 0   
    elif(chest_pain_type == 'Nonanginal' or chest_pain_type=='nonanginal'):
        input_dict["cp_atypical angina"]=0
        input_dict["cp_non-anginal"]=1
        input_dict["cp_typical angina"]=0  
    elif(chest_pain_type == 'Asymptomatic' or chest_pain_type=='asymptomatic'):
        input_dict["cp_atypical angina"]=0
        input_dict["cp_non-anginal"]=0
        input_dict["cp_typical angina"]=0           

    

    # fasting blood suger
    if (fasting_blood_sugar== 'False'):
         input_dict["fbs_True"] = 0
    else:
         input_dict["fbs_True"] = 1 


    # resting ecg
    if(resting_ecg== 'Normal'):
        input_dict["restecg_normal"] = 1
        input_dict["restecg_stt abnormality"] = 0

    elif(resting_ecg== 'sttabnormality'):
        input_dict["restecg_normal"] = 0
        input_dict["restecg_st-t abnormality"] = 1

    elif(resting_ecg== 'iv_Hypertrophy'):
        input_dict["restecg_normal"] = 0
        input_dict["restecg_st-t abnormality"] = 0   
        

    #  exercise-induced angina
    if (exercise_angina=='False'):
        input_dict["exang_True"] = 0
    else:
        input_dict["exang_True"] = 1


    #  st_slope
    if(st_slope == 'flat'):
        input_dict["slope_flat"] = 1
        input_dict["slope_upsloping"] = 0
    if(st_slope == 'upsloping'):
        input_dict["slope_flat"] = 0
        input_dict["slope_upsloping"] = 1
    if(st_slope == 'downsloping'):
        input_dict["slope_flat"] = 0
        input_dict["slope_upsloping"] = 0
            
       
    # Return in the same column order as your training data
        ''''age', 'trestbps', 'chol', 'thalch( max_heart_rate)', 'oldpeak', 'sex_Male',
       'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 'fbs_True',
       'restecg_normal', 'restecg_st-t abnormality', 'exang_True',
       'slope_flat', 'slope_upsloping' '''
    final_features = [
        input_dict["age"],
        input_dict["trestbps"],
        input_dict["chol"],
        input_dict["thalch"],  #max_heart_rate
        input_dict["oldpeak"],
        input_dict["sex_Male"],
        input_dict["cp_atypical angina"],
        input_dict["cp_non-anginal"],
        input_dict["cp_typical angina"],
        input_dict["fbs_True"],
        input_dict["restecg_normal"],
        input_dict["restecg_st-t abnormality"],
        input_dict["exang_True"],
        input_dict["slope_flat"],
        input_dict["slope_upsloping"],
    ]

    return final_features
