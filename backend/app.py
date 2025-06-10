from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="templates"
)

CORS(app, origins=["https://simunex.netlify.app"])

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("ERROR: Missing GROQ_API_KEY.")

llm=None
def get_llm():
    global llm
    if llm is None:
        llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model="llama3-70b-8192"
        )
    return llm

# Import the detection function (new file)
from yolov8_detect import detect_objects

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/detect', methods=['POST'])
def detect_components():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image = request.files['image']
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image.filename)

    try:
        import psutil
        if psutil.virtual_memory().available < 200 * 1024 * 1024:  # 200MB
            return "Server memory overloaded", 503

        image.save(image_path)
        logger.info(f"Image saved to {image_path}")

        detected_components = detect_objects(image_path)
        logger.info(f"Components detected: {detected_components}")

        # Handle no detection
        if not detected_components:
            suggestions = "Could not detect components. Please try a clearer image."
            os.remove(image_path)
            return jsonify({"error": "No components detected", "suggestions": suggestions}), 400

        # Generate suggestions from LLM
        prompt = f"Suggest simple IoT projects using: {', '.join(detected_components)}"
        suggestions = get_llm().invoke(prompt).content

        if os.path.exists(image_path):
            os.remove(image_path)

        return render_template('result.html', 
                               components=detected_components, 
                               suggestions=suggestions)

    except MemoryError:
        return "Memory error - try a smaller image", 413

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}", 500


@app.route('/wokwi')
def wokwi_page():
    return render_template('wokwi.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        response = llm.invoke(prompt).content
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
