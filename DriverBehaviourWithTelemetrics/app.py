from flask import Flask, flash, request, redirect, render_template, Response
import predict
from predict import StackingRegressor
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/Users/Harikrishna/Desktop\LSTM/DetectionTelemetric_LSTM/_tests'
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modelexec',methods=['GET' , 'POST'])
def performmodelexecution():
    print("performmodelexecution")
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return StackingRegressor.predictfn(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)