# 🌱 EcoSnap – AI Food Classifier with Carbon Estimator

EcoSnap is a Streamlit-powered AI application that identifies food items from images and estimates their carbon footprint based on sustainable consumption data. The goal is to raise environmental awareness and promote eco-friendly food choices through visual interaction and AI.

---

## 📸 Features

- 🧠 Deep learning–based food classification using MobileNetV2.
- 📊 Real-time carbon footprint estimation (kg CO₂/kg food).
- 💡 Personalized sustainability tips for each food item.
- ✅ Easy-to-use web app interface via Streamlit.
- 🔁 Extendable dataset and model.

---

## 🗂️ Project Structure

EcoSnap/  
│  
├── app.py # Streamlit web app  
├── train_model.ipynb # Jupyter notebook for model training  
├── requirements.txt # Python dependencies  
│  
├── data/  
│ ├── carbon_data.csv # CSV with food, CO₂ footprint, and tips  
│ └── class_names.txt # Class labels for model output  
│  
├── dataset/ # Image dataset for training (one folder per class)  
│ ├── apple_pie/  
│ ├── caesar_salad/  
│ └── ...  
│  
└── model/  
└── food_model.h5 # Trained Keras model  

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/EcoSnap.git
cd EcoSnap
```
### 2. Set up the environment
We recommend using a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at: http://localhost:8501

## 🧠 Model Training
You can retrain the model using train_model.ipynb:

Place your food images in the dataset/ folder (one subfolder per class).

Run the notebook to generate a new food_model.h5.

## 📦 Dependencies
Python 3.10+
TensorFlow
Streamlit
pandas
numpy
Pillow

See requirements.txt for full details.

## 📈 Carbon Dataset Format
File: data/carbon_data.csv
```csv
Food,CO2_per_kg,Tip
pizza,5.4,"Go veggie for a lower footprint."
apple_pie,3.1,"Use fresh apples, reduce sugar."
...
```
Ensure tips with commas are enclosed in double quotes.

## 🙌 Contributions
Feel free to contribute! Pull requests and ideas to expand the dataset or improve predictions are welcome.

## 📜 License
This project is licensed under the MIT License.

## 👩‍💻 Author
Pranita Dadhe
Built with ❤️ for sustainable tech solutions.
```
Let me know if you want a version with a **screenshot**, **GIF demo**, or **deployment instructions on Streamlit Cloud.
```
