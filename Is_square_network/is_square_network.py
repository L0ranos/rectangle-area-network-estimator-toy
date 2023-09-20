import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

#wybieramy stan losowy układu dla powtarzalności
random_state = 2137
np.random.seed = random_state

#Wygenerujmy sobie długości naszych boków prostokąta
x_lengths = np.linspace(0, 10, 100000)
y_lengths = np.linspace(0, 10, 100000)

squares_training_data = np.vstack((x_lengths,y_lengths)).T

#Przetasujmy sobie wektor, żeby mieć różne długości boków
#Można w pewnym przybliżeniu powiedzieć, że tworzymy teraz prostokąty
#Szansa, że wylosują się identyczne długości jest BARDZO niewielka
np.random.shuffle(y_lengths)
np.random.shuffle(x_lengths)

rectangles_training_data = np.vstack((x_lengths,y_lengths)).T

#Teraz złóżmy kwadraty i prostokąty do jednego wektora
dataset_lengths = np.vstack((squares_training_data, rectangles_training_data))

#Teraz wygenerujemy kategorie 
is_square = dataset_lengths[:,0]==dataset_lengths[:,1]
print(is_square)

#Dzielimy na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(dataset_lengths, is_square, train_size=0.9, random_state=random_state)

#Definiujemy sieć neuronową
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=[2]),
    layers.Dense(5, activation='relu'), 
    layers.Dense(1, activation="sigmoid"),
])

#Kompilujemy sieć z funkcją celu i optymalizatorem
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

#Szkolimy sieć
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=100,
    epochs=6,
)

#Zapiszmy model testowy
model.save("Is_square_network/trained_model")

#sprawdźmy na zbiorze testowym
preds = model.predict(X_test)
preds = preds>=0.5

#Wygenerujmy graficzną macierz pomyłek
conf_mat = confusion_matrix(y_test, preds)
confmat = plt.figure()
plt.matshow(conf_mat)
plt.title("Macierz pomyłek sieci")

for (x, y), value in np.ndenumerate(conf_mat):
    plt.text(x, y, f"{value:.0f}", va="center", ha="center",bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.ylabel('Kategoria rzeczywista')
plt.xlabel('Kategoria przewidziana')
plt.xticks([0,1],["Kwadrat", "Prostokąt"])
plt.yticks([0,1],["Kwadrat", "Prostokąt"], rotation=90)
plt.savefig('Is_square_network/confmat_test_live.png', bbox_inches="tight")

#Zróbmy ładny wykres procesu treningu
history_df = pd.DataFrame(history.history)
histfig = plt.figure()
plt.plot(history_df['loss'])
plt.plot(history_df['val_loss'])
plt.xlabel("Epoka szkolenia")
plt.ylabel("Binarna entropia krzyżowa")
plt.legend(["Strata treningu", "Strata walidacji"])
plt.grid(True, "major")
plt.savefig("Is_square_network/loss_test_live.png")

history_df = pd.DataFrame(history.history)
histfig = plt.figure()
plt.plot(history_df['accuracy'])
plt.plot(history_df['val_accuracy'])
plt.xlabel("Epoka szkolenia")
plt.ylabel("Dokładność klasyfikacji")
plt.legend(["dokładność treningu", "dokładność walidacji"])
plt.grid(True, "major")
plt.savefig("Is_square_network/acc_test_live.png")