import pyrebase

config = {
    "apiKey": "AIzaSyBYZH86Yb04bZnNwtZ4TsO6Rdy4JrBBjVM",
    "authDomain": "iq-53182.firebaseapp.com",
    "databaseURL": "https://iq-53182.firebaseio.com",
    "projectId": "iq-53182",
    "storageBucket": "iq-53182.appspot.com",
    "messagingSenderId": "917387417404"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("armandonarcizoruedaperez@gmail.com", "armando123")
local_id = user["localId"]
id_token = user['idToken']