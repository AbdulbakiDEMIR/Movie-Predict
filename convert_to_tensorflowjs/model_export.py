from tensorflow.keras.models import load_model

# .h5 modelini yükle
model = load_model("movie_predict_model.h5")
print("✅ H5 modeli başarıyla yüklendi.")

# SavedModel formatına dönüştür ve klasöre kaydet
model.export("saved_model_dir")
print("✅ SavedModel formatında klasöre kaydedildi: saved_model_dir")
