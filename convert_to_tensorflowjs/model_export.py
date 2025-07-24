from tensorflow.keras.models import load_model

model = load_model("movie_predict_model.h5")
print("✅ H5 model loaded successfully.")

model.export("saved_model_dir")
print("✅ Saved in SavedModel format to folder: saved_model_dir")