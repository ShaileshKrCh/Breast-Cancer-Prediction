# Breast Cancer Prediction Web App 🧬

Hey there! This is a full-stack web application that predicts whether a breast mass tumor sample is **Malignant** (cancerous) or **Benign** (non-cancerous) based on cell measurements. 

Instead of just giving you a boring text answer, it takes your inputs, checks them against a pre-trained machine learning model, and shows you a beautiful **Radar Chart** and an **Animated Risk Gauge** to visualize the data.

---

## 🚨 Crucial Disclaimer First!
This app is just a cool coding project for educational purposes and portfolio building. It is **NOT** a real medical tool, it has no FDA approval, and it should absolutely never be used to make actual health decisions. Always talk to a real doctor!

---

## ✨ Cool Features Built Into This App
* **Brain Over Brawn (Only 5 Inputs):** The raw dataset actually asks for 30 different medical measurements. Nobody wants to type all that out! I narrowed it down to the top 5 most important ones so the app is fast and easy to use.
* **4-Way Model Battles:** The training script automatically tests four different AI models (Random Forest, Gradient Boosting, Logistic Regression, and SVM) to find out which one gets the highest accuracy on the data.
* **Fair Training (SMOTE):** If a dataset has way more benign cases than malignant ones, models get lazy and just guess "benign" to get a high score. I used SMOTE to artificially balance the data so the model learns both sides perfectly.
* **No Ugly Reloads:** The frontend uses clean JavaScript fetch calls, meaning when you click predict, the graphs animate smoothly on the screen without reloading the whole page.

---

## 📂 Project Structure
```text
breast_cancer_project/
│
├── train.py          # The AI brain setup. Trains the 4 models and saves the best one.
├── app.py            # The Flask backend server that talks between the web page and the AI.
└── templates/
    └── index.html    # The webpage layout, styled with Tailwind CSS and Chart.js graphs.
