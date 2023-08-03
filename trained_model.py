from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    model = load_model(model_path)
    return model
