from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from yolov8_detect import detect_objects
from llm_suggestions import suggest_projects

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="../backend/templates"
)

# Configure CORS - adjust origins as needed
CORS(app, origins=["https://simunex.netlify.app", "http://localhost:5000"])

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
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Save upload
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, secure_filename(request.files['image'].filename))
    request.files['image'].save(image_path)

    try:
        # Run detection (with debug mode for development)
        detected = detect_objects(image_path, debug=True)
        
        if not detected:
            return jsonify({
                "error": "No components detected",
                "debug_tip": "1. Check debug_original.jpg and debug_processed.jpg 2. Try closer/more contrasted shots"
            }), 400
        
        # Generate suggestions
        suggestions = suggest_projects(detected)
        
        return jsonify({
            "components": detected,
            "suggestions": suggestions,
            "memory_used": f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB"
        })

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))