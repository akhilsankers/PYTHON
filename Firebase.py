import firebase_admin
from firebase_admin import db, credentials

try:
    # Authenticate to Firebase
    cred = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(cred, {"databaseURL": "https://thirdeye-7a6d6-default-rtdb.firebaseio.com"})

    # Creating a reference to the root node
    ref = db.reference("/")

    # Initializing title_count before the transaction
    db.reference("/title_count").set(0)

    # Retrieving data from the root node
    data = ref.get()
    if data:
        print(data)
    else:
        print("No data found.")

    # Update operation (add new key 'name' with value 'python')
    db.reference("/").update({"name": "python"})
    print(ref.get())

    # Get value from a specific key ('name' in this case)
    name_value = db.reference("/name").get()
    if name_value is not None:
        print(name_value)
    else:
        print("No value found for key 'name'.")

except firebase_admin.exceptions.FirebaseError as firebase_error:
    print(f"Firebase error: {firebase_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
