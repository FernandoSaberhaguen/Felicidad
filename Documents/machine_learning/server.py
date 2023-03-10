import joblib
import numpy as np

from flask import Flask
from flask import jsonify
app= Flask(__name__)


#Postman pruebas
@app.route('/predict', methods=['GET'])
def predict():
    X_test= np.array([7.594445,7.479556, 1.616463, 1.533524, 0.796667, 0.635423, 0.362012, 0.315964, 2.277027])
    precition= model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion': list(precition)})


if __name__ == '__main__':
    model= joblib.load('./models/best_model.pkl')
    app.run(port=8080)
