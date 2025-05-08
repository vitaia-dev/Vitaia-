
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Charger les données
df = pd.read_csv("parkinsons.data (1).txt")
X = df.drop(columns=["name", "status"])
y = df["status"]

# Préparer l'IA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route("/analyser", methods=["POST"])
def analyser_audio():
    if 'fichier' not in request.files:
        return jsonify({"erreur": "Aucun fichier reçu"}), 400

    fichier_audio = request.files['fichier']
    chemin_temp = "temp_audio.wav"
    fichier_audio.save(chemin_temp)

    try:
        rate, data_audio = wavfile.read(chemin_temp)
        if data_audio.dtype != np.float32:
            data_audio = data_audio.astype(np.float32) / np.max(np.abs(data_audio))

        mean_amp = np.mean(data_audio)
        std_amp = np.std(data_audio)
        max_amp = np.max(data_audio)
        min_amp = np.min(data_audio)

        features = np.array([mean_amp, std_amp, max_amp, min_amp])
        features = np.pad(features, (0, X.shape[1] - len(features)), mode='constant')
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        resultat = "Signes de Parkinson détectés" if prediction[0] == 1 else "Aucun signe détecté"
        return jsonify({"resultat": resultat})

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

    finally:
        if os.path.exists(chemin_temp):
            os.remove(chemin_temp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
