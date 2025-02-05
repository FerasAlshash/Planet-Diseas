from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)


model_path = 'D:/VS Code/Projects/Plant Leaf Disease Detection/plant_disease_classifier_CNN.h5'
model = load_model(model_path)


class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    5: 'Corn_(maize)___Common_rust_',
    6: 'Corn_(maize)___Northern_Leaf_Blight',
    7: 'Corn_(maize)___healthy',
    8: 'Grape___Black_rot',
    9: 'Grape___Esca_(Black_Measles)',
    10: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    11: 'Grape___healthy',
    12: 'Potato___Early_blight',
    13: 'Potato___Late_blight',
    14: 'Tomato___Bacterial_spot',
    15: 'Tomato___Late_blight',
    16: 'Tomato___Septoria_leaf_spot',
    17: 'Tomato___Target_Spot',
    18: 'Tomato___Tomato_mosaic_virus'
}

# دالة معالجة الصورة
def preprocess_image(image):
    try:
        image = image.convert('RGB')  # تحويل إلى RGB
        image = image.resize((150, 150), Image.Resampling.LANCZOS)  
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0) 
        print("Processed Image Shape:", image_array.shape)  
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        try:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            if processed_image is None:
                return jsonify({"error": "Image processing failed"}), 500

            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            label = class_labels.get(predicted_class, "Unknown Disease")

            
            print(f"Prediction Probabilities: {prediction}")
            print(f"Predicted Class Index: {predicted_class}, Label: {label}")

            return jsonify({"label": label, "confidence": float(np.max(prediction))})

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": f"Error processing image: {e}"}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
