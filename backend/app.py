from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)
CORS(app)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("ERROR: Missing GROQ_API_KEY.")

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)

model = YOLO('model/best.pt')

@app.route('/')
def home():
    return render_template('front.html')  

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

@app.route('/detect', methods=['POST'])
def detect_components():
    image = request.files.get('image')
    if not image:
        return "No image uploaded", 400

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image.filename)
    image.save(image_path)

    try:
        results = model(image_path, conf=0.4)

        detected_components = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                component_name = model.names[class_id]
                detected_components.add(component_name)

        detected_components = list(detected_components)
        print(f"[DEBUG] Components detected: {detected_components}")

        if not detected_components:
            detected_components = ['No components detected']
            suggestions = "Could not detect components in the image. Please try uploading a clearer image."
        else:
            prompt = f"Suggest simple IoT projects using: {', '.join(detected_components)}"
            suggestions = llm.invoke(prompt).content

        return render_template('result.html', components=detected_components, suggestions=suggestions)

    except Exception as e:
        return f"An error occurred during detection: {e}", 500

@app.route('/wokwi')
def wokwi_page():
    return render_template('wokwi.html')

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': '⚠️ Please enter a valid question.'}), 400
        response = llm.invoke(prompt)
        return jsonify({'reply': response.content})
    except Exception as e:
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
