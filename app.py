from flask import Flask, render_template, request
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load("model.pkl")
pipeline=joblib.load("pipeline.pkl")

@app.route('/')
def home():
    return render_template("main.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data={
            
            'gender':request.form['gender'],
            'married':request.form['married'],
            'dependants':request.form['dependants'],
            'edu':request.form['edu'],
            'slf_emp':request.form['slf_emp'],
            'income':request.form['income'],
            'co_income':request.form['co_income'],
            'amnt':request.form['amnt'],
            'loan_term':request.form['loan_term'],
            'credit_history':request.form['credit_history'],
            'area':request.form['area']
        }

        input_data={
            'gender':[form_data['gender']],
            'married':[form_data['married']],
            'dependants':[form_data['dependants']],
            'edu':[form_data['edu']],
            'slf_emp':[form_data['slf_emp']],
            'income':[form_data['income']],
            'co_income':[form_data['co_income']],
            'amnt':[form_data['amnt']],
            'loan_term':[form_data['loan_term']],
            'credit_history':[form_data['credit_history']],
            'area':[form_data['area']]
        }

        int_df=pd.DataFrame(input_data)
        processed_data=pipeline.transform(int_df)
        prediction=model.predict(processed_data)[0] # scikit-learn by default predict an array e.g ['y'] instead of just 'y' that's why [0] is used
        result= "Approved" if prediction=='Y' else "Rejected"

        return render_template("main.html", prediction_text=result)
    except Exception as e:
        return render_template("main.html", prediction_text=f"Error : {str(e)}")
    
    

if __name__=="__main__":
    app.run(debug=True)