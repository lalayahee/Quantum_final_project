import joblib, os
model_dir = r'c:\Year 4\Quantum\Quantum_final_project\classical_ml\models'
candidates = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
print('candidates:', candidates)
for name in candidates:
    path = os.path.join(model_dir, name)
    try:
        m = joblib.load(path)
        print(name, 'loaded OK, predict_proba:', hasattr(m, 'predict_proba'))
    except Exception as e:
        print(name, 'ERROR:', type(e), e)
