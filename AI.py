"""
Copyright [2023] [Matteo Raso]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv("data/roboBohr.csv")
data = data.drop(["Unnamed: 0", "pubchem_id"], axis=1)

X = data.drop(["Eat"], axis=1)
Y = data["Eat"]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer="adam", loss="mse")

history = model.fit(X, Y, epochs=100, validation_split=0.2)

model.save("saved_model/initial_model")

plt.plot(history.history["loss"], label="loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot()
plt.show()
