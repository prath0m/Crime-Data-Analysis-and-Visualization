# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Load the training data
# df = pd.read_csv('training.csv', names=['description', 'type'], header=None)

# # Create the TF-IDF vectorizer and transform the training data
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# # Train the SVC model with the RBF kernel
# svm_type_classifier = SVC(kernel='rbf')
# svm_type_classifier.fit(tfidf_matrix, df['type'])

# # Save the trained model
# joblib.dump((svm_type_classifier, tfidf_vectorizer), 'trained_model.joblib')


# import joblib

# # Load the pre-trained model
# svm_type_classifier, tfidf_vectorizer = joblib.load('trained_model.joblib')

# def classify_description(description):
#     tfidf_new = tfidf_vectorizer.transform([description])
#     predicted_type = svm_type_classifier.predict(tfidf_new)[0]
#     return predicted_type

# import re

# def preprocess(text):
#     text = re.sub(r'[^\w\s\']',' ',text)
#     text = re.sub(r' +',' ',text)
#     return text.strip().lower()

# text_predict = """In a quiet suburban neighborhood, a daring burglary unfolded under the cover of darkness. A skilled thief, equipped with tools of the trade, discreetly 
# entered a residence through an unlocked back door. Moving with calculated precision, the intruder swiftly ransacked the home, absconding with valuable jewelry and electronic devices.
# The residents, unaware of the intrusion until morning, were left shocked and violated by the audacious crime that had taken place within the sanctity of their own home. 
# Law enforcement was promptly notified, initiating an investigation into the burglary that left the community on edge."""

# text_predict = preprocess(text_predict)
# text_predict = text_predict.replace("\n", " ")
# # text_predict = "attempted to murder a woman from a car driver"

# predicted_type = classify_description(text_predict)

# # Display the predicted Category and Type
# print(f"Predicted Type: {predicted_type}")




# import pyttsx3


# def Speak(Text):
#     engine = pyttsx3.init("sapi5")
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[0].id)
#     engine.setProperty('rate', 170)
#     print("")
#     print(f"You : {Text}.")
#     print("")
#     engine.say(Text)
#     engine.runAndWait()

# from selenium import webdriver
# from selenium.webdriver.support.ui import Select
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from time import sleep


# chrome_options = Options()
# chrome_options.add_argument('--log-level=3')
# chrome_options.headless = False

# # Update the path to your Chrome executable
# Path = "Database\\chromedriver.exe"
# driver = webdriver.Chrome(Path, options=chrome_options)
# driver.maximize_window()

# website = r"https://ttsmp3.com/text-to-speech/British%20English/"
# driver.get(website)
# ButtonSelection = Select(driver.find_element(by=By.XPATH,value='/html/body/div[4]/div[2]/form/select'))
# ButtonSelection.select_by_visible_text('British English / Brian')
# #Speak("Hello, I am speaking using text-to-speech!")

# def Speak(Text):
    
#     lengthoftext = len(str(Text))

#     if lengthoftext == 0:
#         pass

#     else:
#         print("")
#         print(f"AI : {Text}")
#         print("")
#         Data = str(Text)
#         xpathofsec = '/html/body/div[4]/div[2]/form/textarea'
#         driver.find_element(By.XPATH,value=xpathofsec).send_keys(Data)
#         driver.find_element(By.XPATH,value='//*[@id="vorlesenbutton"]').click()
#         driver.find_element(By.XPATH,value="/html/body/div[4]/div[2]/form/textarea").clear()

#         if lengthoftext>=30:
#             sleep(4)

#         elif lengthoftext>=40:
#             sleep(6)

#         elif lengthoftext>=55:
#             sleep(8)

#         elif lengthoftext>=70:
#             sleep(10)

#         elif lengthoftext>=100:
#             sleep(13)

#         elif lengthoftext>=120:
#             sleep(14)

#         else:
#             sleep(2)


# Speak("Welcome Vedant Sir")

# from pymongo import MongoClient

# # Connect to the MongoDB servers
# # source_client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/")
# # destination_client = MongoClient("mongodb+srv://himanshu:himanshu@cluster0.gaoteru.mongodb.net/")
# source_client = MongoClient("mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/Crime")
# destination_client = MongoClient("mongodb+srv://himanshu:himanshu@cluster0.gaoteru.mongodb.net/Crime")

# # Select the source and destination databases and collections
# source_db = source_client['Crime']
# destination_db = destination_client['Crime']

# source_collection = source_db['CrimeDetails']
# destination_collection = destination_db['CrimeDetails']

# i=1
# # Iterate through each document in the source collection and insert it into the destination collection
# for document in source_collection.find():
#     destination_collection.insert_one(document)
#     print(i)
#     i = i+1

# # Close the MongoDB connections
# source_client.close()
# destination_client.close()


# import pymongo

# # Connect to MongoDB
# client = pymongo.MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/")
# db = client["Crime"]
# collection = db["CrimeDetails"]

# # Update documents
# update_result = collection.update_many(
#     {"Landmark": "Select"},
#     {"$set": {"Landmark": "SELECT"}}
# )

# # Print the number of documents updated
# print(f"Number of documents updated: {update_result.modified_count}")

# # Close the MongoDB connection
# client.close()

# import pymongo
# import random

# # Connect to MongoDB
# client = pymongo.MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
# db = client["Crime"]
# collection = db["CrimeDetails"]

# # List of police stations
# # police_stations = ['BHOSARI', 'BHOSARI MIDC', 'CHIKHALI','PIMPRI','CHINCHWAD','NIGADI','CHAKAN','ALANDI','DIGHI','MHALUNGE','SANGVI','WAKAD','HINJEWADI','RAVET','DEHUROAD','TALEGAON DABHADE','SHIRGAON','TALEGAON MIDC']

# # Update documents with a random police station
# # for document in collection.find():
# #     random_station = random.choice(police_stations)
# #     collection.update_one(
# #         {"_id": document["_id"]},
# #         {"$set": {"Police_Station": random_station}}
# #     )

# # # Print a message indicating the update
# # print(f"Police Station field added to all documents with random values.")
# query = {"Date_of_Crime": {"$regex": "^2018-"}}
# update = {"$set": {"Date_of_Crime": "2024-01-14"}}

# result = collection.update_many(query, update)

# print(f"Matched {result.matched_count} documents and modified {result.modified_count} documents.")
# # Close the MongoDB connection
# client.close()
# from pymongo import MongoClient 
# import ssl

# def copy_collection(source_uri, destination_uri, source_db, destination_db, collection_name, dest_coll):
#     # Connect to the source MongoDB
#     # source_client = MongoClient(source_uri)
#     source_client = MongoClient(source_uri)

#     source_database = source_client[source_db]
#     source_collection = source_database[collection_name]

#     # Connect to the destination MongoDB
#     destination_client = MongoClient(destination_uri)
#     destination_database = destination_client[destination_db]
#     destination_collection = destination_database[dest_coll]

#     # Copy documents from source collection to destination collection
#     destination_collection.insert_many(source_collection.find())     

#     # Close connections
#     source_client.close()
#     destination_client.close()

# if __name__ == "__main__":
#     # Specify your source and destination MongoDB URIs
#     source_mongodb_uri = "mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority"
#     destination_mongodb_uri = "mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority"
    
#     # source_mongodb_uri = "mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority"
#     # destination_mongodb_uri = "mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority"

#     # Specify the source and destination databases and collection name
#     source_database_name = "Crime1"
#     destination_database_name = "Crime"
#     collection_to_copy = "UserDetails1"
#     dest_coll = "UserDetails"

#     # Copy the collection
#     copy_collection(source_mongodb_uri, destination_mongodb_uri, source_database_name, destination_database_name, collection_to_copy,dest_coll)

#     print(f"Collection '{collection_to_copy}' copied from '{source_mongodb_uri}' to '{destination_mongodb_uri}'.")



# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority')  # Update the connection string with your MongoDB URI
# db = client['Crime']  # Replace 'your_database_name' with your actual database name
# collection = db['CrimeDetails']  # Replace 'your_collection_name' with your actual collection name

# # Retrieve all records from the collection
# all_records = list(collection.find())

# # Identify the records to keep (first 96 records)
# records_to_keep = all_records[:96]

# # Delete the remaining records from the collection
# for record in all_records[96:]:
#     collection.delete_one({'_id': record['_id']})

# print("Remaining records after deletion:", collection.count_documents({}))



from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority') 
database = client['Crime']  # Replace 'your_database_name' with your actual database name
collection = database['CrimeDetails']  # Replace 'your_collection_name' with your actual collection name

# Get all documents from the collection
documents = collection.find()

# Update the firno field for each document
for index, document in enumerate(documents, start=1):
    # Update the firno field with the incremented value
    collection.update_one(
        {'_id': document['_id']},  # Assuming '_id' is the unique identifier field
        {'$set': {'FIR_No': index}}
    )
    print(document['_id'])

print('Firno field updated successfully.')

# Close the MongoDB connection
client.close()


















# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from time import sleep
# def get_marks(id_code):
#     # Initialize Selenium webdriver
#     driver = webdriver.Chrome()  # You need to download the appropriate WebDriver for your browser

#     # Open the website
#     driver.get("https://gpamravati.ac.in/result/")

#     try:
#         # Get the CAPTCHA text
#         captcha_text = driver.find_element(by=By.XPATH, value='//*[@id="CaptchaDiv"]').text
#         captcha_text = captcha_text.strip()
#         print("CAPTCHA:", captcha_text)

#         # Find and enter ID code
#         id_element = driver.find_element(by=By.XPATH, value='/html/body/table/tbody/tr[4]/td/table/tbody/tr[3]/td[2]/table/tbody/tr[2]/td[2]/input')
#         id_element.send_keys(id_code)

#         # Find and enter CAPTCHA
#         captcha_input = driver.find_element(by=By.XPATH, value='//*[@id="CaptchaInput"]')
#         captcha_input.send_keys(captcha_text)

#         # Submit the form
#         submit_button = driver.find_element(by=By.XPATH, value='/html/body/table/tbody/tr[4]/td/table/tbody/tr[3]/td[2]/table/tbody/tr[5]/td/div/input')
#         submit_button.click()

#         driver.execute_script("""
#             var form = document.createElement('form');
#             form.method = 'post';
#             form.action = 'result.php';  // URL where you want to submit the form
#             var input = document.createElement('input');
#             input.type = 'hidden';
#             input.name = 'regno';  // Name of the parameter
#             input.value = '%s'; // Value of the parameter
#             form.appendChild(input);
#             var inputa = document.createElement('input');
#             inputa.type = 'hidden';
#             inputa.name = 'ucap_text';  // Name of the parameter
#             inputa.value = '%s'; // Value of the parameter
#             form.appendChild(inputa);
#             document.body.appendChild(form);
#             form.submit();
#         """ % (id_code, captcha_text))
#         marks1 = driver.find_element(by=By.XPATH, value='/html/body/table/tbody/tr/td/table/tbody/tr[4]/td/table/tbody/tr[2]/td[3]').text
#         print("MARKS1:", marks1)





#     except Exception as e:
#         print('Exception:', e)

#     finally:
#         # Close the browser
        
#         print(driver.current_url)
#         driver.quit()

# if __name__ == "__main__":
#     get_marks('21CM061')
