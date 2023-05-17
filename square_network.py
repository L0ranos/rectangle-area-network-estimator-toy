import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import pandas as pd

#wybieramy stan losowy układu dla powtarzalności
random_state = 42
np.random.seed = random_state


#Wygenerujmy sobie długości naszych boków prostokąta
x_lengths = np.linspace(0, 10, 100000)
y_lengths = np.linspace(0, 10, 100000)
#Przetasujmy sobie wektor, żeby mieć różne długości boków
np.random.shuffle(y_lengths)
np.random.shuffle(x_lengths)

print(np.shape(y_lengths))
#Teraz policzmy pole
square_area = x_lengths*y_lengths

#Dzielimy na zbiór uczący i testowy
Training_data = np.vstack((x_lengths,y_lengths)).T
print(np.shape(Training_data))

X_train, X_test, y_train, y_test = train_test_split(Training_data, square_area, train_size=0.9)

#Definiujemy sieć neuronową
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=[2]),
    layers.Dense(5, activation='relu'), 
    layers.Dense(1),
])

#Kompilujemy sieć z funkcją celu i optymalizatorem
model.compile(
    optimizer='adam',
    loss='mae'
)
model.summary()

#Szkolimy sieć
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=10,
    epochs=12,
)

#Zapiszmy model testowy
model.save("trained_model_live")

#sprawdźmy na zbiorze testowym
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print("_______________________________________")
print(f"Testowy średni błąd bezwzględny: {mae}")

#Zróbmy ładny wykres procesu treningu
history_df = pd.DataFrame(history.history)
plt.plot(history_df['loss'])
plt.plot(history_df['val_loss'])
plt.xlabel("Epoka szkolenia")
plt.ylabel("Średni błąd bezwzględny")
plt.legend(["Strata treningu", "Strata walidacji"])
plt.grid(True, "major")
plt.savefig("result_test_live.png")