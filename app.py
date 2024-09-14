import json
from flask import *
import pandas as pd
import numpy as np
import os
from PIL import Image
from flask_session import Session
import random
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import csv
import math
from geopy.geocoders import Nominatim

app = Flask(__name__)
model = tf.keras.models.load_model("plant_identification_model2.keras")
main_data_dir = "static/image/Leaf Images"

# Create label mapping based on subdirectory names
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}


def disease_prediction(Symptom1, Symptom2, Symptom3, Symptom4, Symptom5):
    # Import necessary libraries
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score

    # Load the dataset and other necessary data
    with open('data/symptoms.json', 'r') as json_file:
        dataset = json.load(json_file)
    l1 = dataset.get("name", [])

    # Load the testing data
    tr = pd.read_csv("data/Testing.csv")

    # Load the prognosis datas
    with open('data/prognosis.json', 'r') as json_file:
        dataset = json.load(json_file)
    prog = {key: int(value) for key, value in dataset.get("prognosis", {}).items()}
    prognosis = {"prognosis": prog}
    tr.replace(prognosis, inplace=True)

    X_test = tr[l1]

    # Load the training data
    df = pd.read_csv("data/Training.csv")
    df.replace(prognosis, inplace=True)

    X = df[l1]

    y = df[["prognosis"]]
    np.ravel(y)

    # Initialize and fit the Naive Bayes model
    gnb = MultinomialNB()
    gnb = gnb.fit(X, np.ravel(y))

    # Predict the disease based on user input symptoms
    psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

    l2 = [0] * len(l1)

    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predicted = gnb.predict(inputtest)[0]

    with open('data/disease_name.json', 'r') as json_file:
        dataset = json.load(json_file)

    diseases = dataset.get("disease",[])

    with open('data/doctors.json', 'r') as file:
        suggested_doctors = json.load(file)

    # Get the predicted disease and suggested doctor
    predicted_disease = diseases[predicted]
    suggested_doctor = suggested_doctors.get(predicted_disease, {"name": "Doctor information not available", "profile_link": ""})

    return predicted_disease, suggested_doctor

# Function to save user data to the JSON file
def save_users(users):
    with open('data/users.json', 'w') as json_file:
        json.dump(users, json_file)

# Function to load user data from the JSON file
def load_users():
    try:
        with open('data/users.json', 'r') as json_file:
            users = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):  # Catch a broader JSONDecodeError exception
        users = {}
    return users

# Load user data initially
users = load_users()

@app.route('/log')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    # csrf_token = request.form.get('csrf_token')
    users = load_users()

    if username in users and users[username]['password'] == password:
        # Successful login
        name = users[username]['name']  # Retrieve user's name
        email = users[username]['email'] 
        return redirect('/')
        
    else:
        return redirect('/log')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')  # Retrieve name
        email = request.form.get('email')  # Retrieve email
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            return 'Username already exists. Please choose a different username.'
        else:
            users[username] = {
                'password': password,
                'name': name,  # Store name
                'email': email  # Store email
            }
            save_users(users)
            
            return redirect('/')

    return render_template('register.html')


@app.route('/home')
def home():
    return render_template('home.html')

#------------------------------------------------------------------------------------------------------------------>

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        disease = request.form['disease']
        with open('data/disease_data1.json', 'r') as json_file:
            disease_symptoms = json.load(json_file)
        if disease in disease_symptoms:
            symptoms = disease_symptoms[disease]
        else:
            symptoms = ["Disease not found."]
        return render_template('symptoms.html', symptoms=symptoms)

    return render_template('symptoms.html', symptoms=None)

@app.route('/get_suggestions')
def get_suggestions():
    query = request.args.get('query')
    with open('data/disease_data1.json', 'r') as json_file:
        disease_symptoms = json.load(json_file)
    suggestions = sorted(list(disease_symptoms.keys()))
    filtered_suggestions = [suggestion for suggestion in suggestions if query.lower() in suggestion.lower()]
    return jsonify(filtered_suggestions)

#--------------------------------------------------------------------------------------------------------------------->

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve user inputs from the form
        symptom1 = request.form.get('symptom1')
        symptom2 = request.form.get('symptom2')
        symptom3 = request.form.get('symptom3')
        symptom4 = request.form.get('symptom4')
        symptom5 = request.form.get('symptom5')

        # Call your prediction function (disease) with user inputs
        # Replace this with your actual prediction logic
        predicted_disease, suggested_doctor = disease_prediction(symptom1, symptom2, symptom3, symptom4, symptom5)

        # Render the results.html template with prediction results
        return render_template('results.html', disease=predicted_disease, doctor=suggested_doctor)

#---------------------------------------------------------------------------------------------------------------------->

# Global data list to store donor information
data = []

# Function to read data from CSV
def read_data_from_csv():
    global data
    data = []
    with open('data/bloodbank.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            data.append(row)

# Function to write data to CSV
def write_data_to_csv():
    with open('data/bloodbank.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Blood Group', 'Phone', 'BloodBank Name', 'Longitude', 'Latitude'])
        writer.writerows(data)

# Function to add a donor
def add_donor(name, blood_group, phone, bloodbank_name, longitude, latitude):
    data.append([name, blood_group, phone, bloodbank_name, longitude, latitude])
    write_data_to_csv()
    # data.append([name, blood_group, phone, bloodbank_name, longitude, latitude])
    # write_data_to_csv()
    # messagebox.showinfo('Success', 'Donor added successfully.')

# Function to search donors by blood group
def search_donor_by_blood_group(blood_group):
    matching_donors = [donor for donor in data if donor[1] == blood_group]
    return matching_donors

@app.route('/bloodbank')
def bloodbank():
    return render_template('bloodbank.html')

@app.route('/add_donor', methods=['GET', 'POST'])
def add_donor_route():
    bloodbank_names= sorted(["PEERLESS HOSPITAL","SSKM HOSPITAL BLOOD BANK","FORTIS HOSPITALS LIMITED BLOOD BANK",\
                            "SOUTH EAST KOLKATA MANAB KALYAN","RUBY GENERAL HOSPITAL","LIONS BLOOD BANK","IBTM&IH",\
                            "OM BLOOD BANK KOLKATA","LIFE CARE BLOOD BANK","BHORUKA BLOOD BANK","PEOPLE'S BLOODBANK"])
    bloodbanks=[
        {"name": "PEERLESS HOSPITAL", "longitude": "88.3939707", "latitude": "22.4810427"},
        {"name": "FORTIS HOSPITALS LIMITED BLOOD BANK", "longitude": "88.3990948", "latitude": "22.5181021"},
        {"name": "SOUTH EAST KOLKATA MANAB KALYAN", "longitude": "88.391084", "latitude": "22.5256993"},
        {"name": "RUBY GENERAL HOSPITAL", "longitude": "88.402884", "latitude": "22.5131759"},
        {"name": "LIONS BLOOD BANK", "longitude": "88.2869754", "latitude": "22.5010317"},
        {"name": "IBTM&IH", "longitude": "88.3751871", "latitude": "22.5854362"},
        {"name": "OM BLOOD BANK KOLKATA", "longitude": "88.3794252", "latitude": "22.5638132"},
        {"name": "LIFE CARE BLOOD BANK", "longitude": "88.370838", "latitude": "22.5498049"},
        {"name": "BHORUKA BLOOD BANK", "longitude": "88.3571161", "latitude": "22.5550749"},
        {"name": "PEOPLE'S BLOODBANK", "longitude": "88.3462425", "latitude": "22.5257451"},
        {"name": "SSKM HOSPITAL BLOOD BANK", "longitude": "88.3437482", "latitude": "22.5402242"}
    ]
    # Get form data
    if request.method == 'GET':
        # Render the add_donor.html template for GET requests
        return render_template('add_donor.html', bloodbank_names=bloodbank_names, bloodbanks=bloodbanks)
    
    name = request.form.get('name')
    blood_group = request.form.get('blood_group')
    phone = request.form.get('phone')
    bloodbank_name = request.form.get('bloodbank_name')
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')

    # Check if all required fields are filled out (you can add more validation)
    if not name or not blood_group or not phone or not bloodbank_name or not longitude or not latitude:
        # Handle the error appropriately, e.g., by rendering an error page
        return render_template('error.html', message='All fields are required.')

    # Assuming you have a function 'add_donor' to add donor data to your system
    add_donor(name, blood_group, phone, bloodbank_name, longitude, latitude)

    # Redirect to a success page after adding the donor
    return redirect(url_for('success'))

@app.route('/success')
def success():
    return render_template('success.html')


@app.route('/search_donor', methods=['GET', 'POST'])
def search_donor_route():
    if request.method == 'POST':
        blood_group = request.form['blood_group']
        
        # Get user's location
        user_location = request.form['location']

        # Geocode the location
        geolocator = Nominatim(user_agent="MyApp")
        location = geolocator.geocode(user_location)

        if location:
            user_lat = location.latitude
            user_lon = location.longitude
            matching_donors = search_donor_by_blood_group(blood_group)
            bloodbanks = {}

            # Group donors by bloodbank
            for donor in matching_donors:
                bloodbank_name = donor[3]
                if bloodbank_name not in bloodbanks:
                    bloodbanks[bloodbank_name] = {'distance': None, 'donors': []}
                bloodbanks[bloodbank_name]['donors'].append(donor)

            # Calculate distances and store bloodbank info
            for bloodbank_name, bloodbank_info in bloodbanks.items():
                bloodbank_lat = float(bloodbank_info['donors'][0][5])
                bloodbank_lon = float(bloodbank_info['donors'][0][4])
                distance = haversine(user_lat, user_lon, bloodbank_lat, bloodbank_lon)
                bloodbank_info['distance'] = distance

            # Sort bloodbanks by distance
            sorted_bloodbanks = sorted(bloodbanks.items(), key=lambda x: x[1]['distance'])
            
            return render_template('bb_results.html', bloodbanks=sorted_bloodbanks)
        else:
            return render_template('location_not_found.html')
    else:
        return render_template('search.html')

# Function to calculate haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

#-------------------------------------------------------------------------------------------------------------->

from sklearn.naive_bayes import MultinomialNB
# Load data outside the function to avoid reloading on every call
with open('data/symptoms.json', 'r') as json_file:
    symptom_data = json.load(json_file)
l1 = symptom_data.get("name", [])

tr = pd.read_csv("data/Testing.csv")
with open('data/prognosis.json', 'r') as json_file:
    prognosis_data = json.load(json_file)
prog = {key: int(value) for key, value in prognosis_data.get("prognosis", {}).items()}
prognosis = {"prognosis": prog}
tr.replace(prognosis, inplace=True)
X_test = tr[l1]

df = pd.read_csv("data/Training.csv")
df.replace(prognosis, inplace=True)
X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Initialize and fit the Naive Bayes model
gnb = MultinomialNB()
gnb = gnb.fit(X, np.ravel(y))

with open('data/disease_name.json', 'r') as json_file:
    disease_data = json.load(json_file)
diseases = disease_data.get("disease", [])

# suggested doctors
with open('data/doctors.json', 'r') as file:
    suggested_doctors = json.load(file)

# Variables to keep track of the conversation state
waiting_for_symptoms = False
symptoms = []
waiting_for_disease = False
disease = []

def disease_prediction_bot(symptoms):
    try:
        psymptoms = symptoms
        l2 = [0] * len(l1)

        for k in range(len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predicted = gnb.predict(inputtest)[0]
        predicted_disease = diseases[predicted]
        suggested_doctor = suggested_doctors.get(predicted_disease, {"name": "Doctor information not available", "profile_link": ""})
        return predicted_disease, suggested_doctor
    
    except Exception as e:
        return "Error in prediction", {"name": str(e), "profile_link": ""}


def get_bot_response(message):
    global waiting_for_symptoms, symptoms, waiting_for_disease, disease
    
    
    message = message.lower()
    
    greetings = {
        r"hi|hello|hey": ["Hello!", "Hi there!", "Hey!", "How're you doing today?How may I help you today?"],
        r"good morning": ["Good morning! How can I help you?", "Morning! What can I do for you?", "Good morning! How can I assist?", "Good Morning!Hope you're doing well"],
        r"good night": ["Good night! Take care!", "Night! Sweet dreams!", "Good night! See you tomorrow!", "Take care!Hope I could have been of some help!"],
        r"bye|goodbye": ["Goodbye! Have a great day!", "Bye! Take care!", "Goodbye! See you soon!", "Adios amigos!"],
        r"how can i help you|how can i assist you": ["I am here to assist you. What do you need help with?", "How can I assist you today?", "What can I do for you?", "Let me know what you're looking for in particular"],
        r"thanks|thank you": ["Welcome...","Mention not","Always there for you...", "Stay Happy, Stay Healthy..."]
    }

    if waiting_for_symptoms:
        if message == "done":
            if len(symptoms) == 0:
                return "You haven't provided any symptoms. Please provide at least one symptom."
            predicted_disease, suggested_doctor = disease_prediction_bot(symptoms)
            waiting_for_symptoms = False
            symptoms = []
            doctor_name = suggested_doctor.get('name', 'Doctor information not available')
            profile_link = suggested_doctor.get('profile_link', 'N/A')
            return (f"The predicted disease is {predicted_disease}.\n"
                    f"Suggested doctor: {doctor_name}\n"
                    f"Profile link: <a href='{profile_link}' target='_blank'>Click Here</a>")
        else:
            if len(symptoms) < 5:
                symptoms.append(message)
                return f"Got it. Please provide another symptom or type 'done' to finish. Symptoms received: {len(symptoms)}"
            else:
                return "You have already provided the maximum number of symptoms. Please type 'done' to finish."

    if waiting_for_disease:
        waiting_for_disease = False
        disease = message
        with open('data/disease_data.json', 'r') as json_file:
            disease_symptoms = json.load(json_file)
        if disease in disease_symptoms:
            symptoms = disease_symptoms[disease]
            return f"The symptoms for {disease} are: {', '.join(symptoms)}"
        else:
            return "Disease not found. Please try another disease name."
    
    for pattern, responses in greetings.items():
        if re.search(pattern, message):
            return random.choice(responses)
    
    if re.search(r"disease|disease prediction", message):
        waiting_for_symptoms = True
        symptoms = []
        return "Tell us your top 5 symptoms one by one. Enter 'done' if no more symptoms left. Enter your first symptom."
    
    if re.search(r"symptom", message):
        waiting_for_disease = True
        disease = []
        return "Please provide the name of the disease to get its symptoms."
    
    return 'Sorry, I did not understand that. Please try again.'


@app.route('/', methods=['GET', 'POST'])
def ayurvedic():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            return redirect(request.url)
        if image:
            # Save the uploaded image temporarily
            image_path = os.path.join('static', 'temp.jpg')
            image.save(image_path)
            return redirect(url_for('predicted', image_path=image_path))
    return render_template('ayurvedic.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            return redirect(request.url)
        if image:
            # Save the uploaded image temporarily
            image_path = os.path.join('static', 'temp.jpg')
            image.save(image_path)
            return redirect(url_for('predicted', image_path=image_path))
    return render_template('upload.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    bot_response = get_bot_response(user_input)
    return jsonify({'response': bot_response})

@app.route('/predict/<image_path>')
def predicted(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]
    return render_template('predicted_results.html', predicted_label=predicted_label, confidence=confidence, image_path=image_path)

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    read_data_from_csv()
    app.run(debug=True)
