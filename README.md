# Quantum ML Real Estate Price Predictor

A Streamlit web application comparing classical machine learning (Random Forest) with quantum machine learning (VQC - Variational Quantum Classifier) for real estate price prediction.

## Features

- **Hybrid ML Comparison**: Compare traditional ML vs Quantum ML predictions
- **Interactive Interface**: Input real estate features and get instant predictions
- **Visualization**: Charts and maps showing prediction confidence
- **Educational**: Learn about quantum machine learning applications

## Local Development

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt

# Run the app
streamlit run frontend/app.py
```

The app will be available at `http://localhost:8501`

## Deployment Options

### 1. Streamlit Cloud (Recommended - Free & Easy)

1. **Prepare your repository**:
   - Ensure all files are committed to Git
   - Make sure `requirements.txt` is up to date
   - Create a `packages.txt` if you have system dependencies (optional)

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the main file path to `frontend/app.py`
   - Click Deploy!

3. **Access your app**:
   - Streamlit Cloud will provide a public URL
   - Your app will be live and accessible worldwide

### 2. Docker Deployment

For more control over the deployment environment, use Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t quantum-ml-app .
docker run -p 8501:8501 quantum-ml-app
```

### 3. Other Cloud Platforms

- **Heroku**: Create a `Procfile` with `web: streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0`
- **AWS/GCP/Azure**: Use their app hosting services with Docker containers

## Project Structure

```
Quantum_final_project/
├── frontend/
│   └── app.py                 # Main Streamlit application
├── quantum_ml/
│   ├── vqc_circuit.py         # Quantum Variational Classifier
│   ├── quantum_preprocess.py  # Quantum data preprocessing
│   ├── trained_params.npy     # Trained quantum parameters
│   └── models/                # Quantum model files
├── classical_ml/
│   ├── models/                # Trained classical ML models
│   ├── data/                  # Training data and features
│   └── *.py                   # ML training and evaluation scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Technologies Used

- **Frontend**: Streamlit
- **Classical ML**: scikit-learn (Random Forest)
- **Quantum ML**: Qiskit, PennyLane (VQC)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, matplotlib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

This project is for educational purposes.