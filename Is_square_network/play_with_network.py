from tensorflow import keras

model = keras.models.load_model("Is_square_network/trained_model")

invalue_x = 0
invalue_y = 0
while True:
    print("____________________________")
    invalue_x = float(input("Długość pierwszego boku:"))
    invalue_y = float(input("Długość drugiego boku:"))
    if invalue_x<0 or invalue_y<0: break

    predicted = model.predict([[invalue_x, invalue_y]])[0][0]
    if predicted>=0.5: predicted="Kwadrat" 
    else: predicted="Prostokąt"
    print(f"Według sieci prostokąt o bokach {invalue_x:.2f}, {invalue_y:.2f} to: {predicted} \n")


