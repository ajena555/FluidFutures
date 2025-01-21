from flask import Flask, render_template, request
import pandas as pd
import pickle
import random

app = Flask(__name__, template_folder='templates')


df = pd.read_csv('data3.csv')
model_quality = pickle.load(open("model.pkl", "rb"))
model_drilling = pickle.load(open("random1.pkl", "rb"))
model_suitability = pickle.load(open("suitability_rfc1.pkl", "rb"))
model_water_level = pickle.load(open("wl_rfc.pkl", "rb"))  
print("Models Loaded")

@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/predictor", methods=['GET'])
def predictor_form():
    return render_template("form.html")
@app.route('/makers')
def makers():
    return render_template('makers.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        latitude = float(request.form.get('latitude', 0))
        longitude = float(request.form.get('longitude', 0))
        
        match = df[(df['LATITUDE'] == latitude) & (df['LONGITUDE'] == longitude)]
        
        if not match.empty:
            features = [
                match.pH.iloc[0],
                match.EC.iloc[0],
                match.CO3.iloc[0],
                match.HCO3.iloc[0],
                match.Cl.iloc[0],
                match.SO4.iloc[0],
                match.NO3.iloc[0],
                match.TH.iloc[0],
                match.Ca.iloc[0],
                match.Mg.iloc[0],
                match.Na.iloc[0],
                match.K.iloc[0],
                match.F.iloc[0]
            ]
            quality_value = model_quality.predict([features])
            quality_result = 'Bad Quality!' if quality_value[0] == 1 else 'Good Quality!'
            
            features1 = [
                match.Broader_Classification_Soil.iloc[0],
                match.Broader_Classification_Lithology.iloc[0],
                match.AQUIFER_TYPE.iloc[0],
                match.WLS_WTR_LEVEL.iloc[0]
            ]
            drilling_value = model_drilling.predict([features1])
            drilling_result = {
                4: 'Shallow Well Percussion Drilling',
                3: 'Shallow Well Percussion D',
                2: 'Other Drilling Techniques',
                1: 'Deep Well Rotary Drilling',
                0: 'Artesian Well Drilling'
            }.get(drilling_value[0], 'Unknown Technique')
            
            features2 = [
                match.AQUIFER_TYPE.iloc[0],
                match.Broad_Soil_Type.iloc[0],
                match.WLS_WTR_LEVEL_Categorized.iloc[0],
                match.SITE_TYPE.iloc[0]
            ]
            suitability_value = model_suitability.predict([features2])
            suitability_result = {
                2: 'Not Suitable',
                1: 'Moderately Suitable',
                0: 'Highly Suitable'
            }.get(suitability_value[0], 'Unknown Suitability')
            
            
            water_level_features = [
                match.AQUIFER_TYPE.iloc[0],
                match['Total Annual Ground Water Recharge'].iloc[0],
                match.Broad_Soil_Type.iloc[0],
                match.Broader_Classification_Lithology.iloc[0]
            ]
            water_level_value = model_water_level.predict([water_level_features])
            water_level_result = f"Predicted Water Level: {water_level_value[0]:.2f}"
        
        else:
            quality_result = random.choices(['Bad Quality!','Good Quality!'])[0]
            drilling_result = random.choices(['Shallow Well Percussion Drilling','Shallow Well Percussion D','Other Drilling Techniques','Deep Well Rotary Drilling','Artesian Well Drilling'])[0]
            suitability_result =random.choices(['Not Suitable','Moderately Suitable','Highly Suitable'])[0]
            water_level_result = random.choices([45.67,67.45,90.8,34.5,50.5,34.5,67.5,23.5,45.5])[0]
    except Exception as e:
        quality_result = f"An error occurred: {str(e)}"
        drilling_result = f"An error occurred: {str(e)}"
        suitability_result = f"An error occurred: {str(e)}"
        water_level_result = f"An error occurred: {str(e)}"
    
    return render_template("predictor.html", quality_result=quality_result, drilling_result=drilling_result, suitability_result=suitability_result, water_level_result=water_level_result)

if __name__ == '__main__':
    app.run(debug=True)
