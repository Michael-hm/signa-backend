from pathlib import Path
import tensorflow as tf
from keras.layers import TFSMLayer
from keras import Model, Input

WINDOW_SIZE = 40
FEATURES = 63

model_path = Path("artifacts") / "signa_model"

print("Path:", model_path.resolve())
print("Exists:", model_path.exists())

# Cargar SavedModel como capa
layer = TFSMLayer(
    str(model_path),
    call_endpoint="serving_default"
)

# UN SOLO Input
inputs = Input(shape=(WINDOW_SIZE, FEATURES))
outputs = layer(inputs)

model = Model(inputs=inputs, outputs=outputs)

print("MODEL LOADED OK")