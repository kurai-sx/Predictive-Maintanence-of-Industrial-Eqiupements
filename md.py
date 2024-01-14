from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model.h5')
inps = np.array([1, 298, 300, 1230, 43, 1])
inps.resize(1, 6)
def answer(l):
    l = np.array(l) 
    l.resize(1, 6)
    c = model.predict(l)
    return c.max()
