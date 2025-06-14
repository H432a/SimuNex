from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename  
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
import psutil 
from yolov8_detect import detect_objects, model  # Importing model from yolov8_detect
from llm_suggestions import suggest_projects, llm
import gc

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="../backend/templates"
)

CORS(app, origins=["https://simunex.netlify.app", "http://localhost:5000"])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # Check file type
    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Secure the filename and save
    try:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    try:
        # Run detection and get results with confidence scores
        results = model(image_path, conf=0.4)
        
        # Process detections to get components with actual confidence scores
        detected_components = []
        for result in results:
            for box in result.boxes:
                detected_components.append({
                    "name": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]) * 100, 1)  # Convert to percentage with 1 decimal
                })

        # Remove duplicate components, keeping the one with highest confidence
        unique_components = {}
        for comp in detected_components:
            name = comp["name"]
            if name not in unique_components or comp["confidence"] > unique_components[name]["confidence"]:
                unique_components[name] = comp

        # Generate suggestions using only component names
        component_names = [comp["name"] for comp in unique_components.values()]
        suggestions = "No components detected"
        if component_names:
            try:
                suggestions = suggest_projects(component_names)
            except Exception as e:
                suggestions = f"Could not generate suggestions: {str(e)}"
                logger.error(f"LLM error: {str(e)}")

        # Return results with actual confidence values
        return render_template(
            'result.html',
            components=list(unique_components.values()),
            suggestions=suggestions
        )

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    finally:
        # Guaranteed cleanup
        if 'image_path' in locals() and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {str(e)}")
        gc.collect()

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        response = llm.invoke(prompt).content
        gc.collect()
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))