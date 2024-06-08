from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import predict_image, convert_to_ela_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 1024 * 1024 * 10  # 10 MB max file size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        ela_image = convert_to_ela_image(filepath, 90)
        ela_image.save(os.path.join('static', 'ela_image.png'))
        
        prediction, confidence = predict_image(filepath)

        return render_template('result.html', prediction=prediction, confidence=confidence, image_path='static/ela_image.png')

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=false,host='0.0.0.0')
