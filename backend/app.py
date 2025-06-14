from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename  
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
import psutil 
from yolov8_detect import detect_objects
from llm_suggestions import suggest_projects,llm
import sys
import platform
import time
from ultralytics import YOLO
from groq import Groq

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

model = YOLO('model/best.pt')
app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="../backend/templates"
)

CORS(app, origins=["https://simunex.netlify.app", "http://localhost:5000"])


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/problem<int:problem_id>')
def problem(problem_id):
    return render_template(f'problems/problem{problem_id}.html')

@app.route('/labpro')
def labpro():
    return render_template('labpro.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/lab')
def lab():
    return render_template('lab.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/wokwi')
def wokwi_page():
    return render_template('wokwi.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')

@app.route('/detect', methods=['POST'])
def detect_components():
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    
    # Check if file is selected
    if image.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Create uploads directory if it doesn't exist
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    
    # Secure the filename and save
    try:
        filename = secure_filename(image.filename)
        image_path = os.path.join(upload_folder, filename)
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    try:
        # Run detection with confidence threshold
        results = model(image_path, conf=0.4)
        
        # Extract unique component names
        detected_components = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_components.add(model.names[class_id])

        detected_components = list(detected_components)
        
        # Generate suggestions if components found
        if not detected_components:
            suggestions = "No components detected. Try a clearer image with better lighting."
        else:
            prompt = f"Suggest 3 simple IoT projects using these components: {', '.join(detected_components)}"
            try:
                suggestions = llm.invoke(prompt).content
            except Exception as e:
                suggestions = f"Could not generate suggestions: {str(e)}"
                logger.error(f"LLM error: {str(e)}")

        # Clean up the uploaded file
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")

        # Return results to template
        return render_template(
            'result.html',
            components=detected_components,
            suggestions=suggestions
        )

    except Exception as e:
        # Clean up file if error occurs
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
                
        logger.error(f"Detection error: {str(e)}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        # Use the suggest_projects function for component-related queries
        if "component" in prompt.lower() or "project" in prompt.lower():
            # Extract components from prompt if possible
            detected_components = []  # You might want to parse components from prompt
            response = suggest_projects(detected_components)
        else:
            # Fall back to general LLM response
            from llm_suggestions import llm
            response = llm.invoke(prompt).content

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"LLM error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    # In a real app, you'd typically get this from a database or session
    # For now, we'll expect the frontend to handle displaying the data
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
