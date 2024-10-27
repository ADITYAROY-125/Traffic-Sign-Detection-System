from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from Detect import detect_objects  # Import the function from detect.py

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename to prevent any directory traversal attacks
            filename = secure_filename(file.filename)
            # Save the file to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Perform object detection on the uploaded image
            results = detect_objects(file_path)  # Call the function from detect.py
            # Process the results as needed
            # Redirect to the results page
            return redirect(url_for('results'))
    return render_template('index.html', error='Invalid file format')

@app.route('/results')
def results():
    # Process the results and display them
    # You can pass any necessary data to the template
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
