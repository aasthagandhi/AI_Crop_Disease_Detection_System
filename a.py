from tensorflow.keras.models import load_model

model = load_model("model/aasu_crop_model.keras")
print(model.output_shape)