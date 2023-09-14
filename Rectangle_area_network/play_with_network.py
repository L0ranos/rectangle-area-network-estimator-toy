from tensorflow import keras

model = keras.models.load_model("Rectangle_area_network/trained_model")

invalue_x = 0
invalue_y = 0
while True:
    print("____________________________")
    invalue_x = float(input("Długość pierwszego boku:"))
    invalue_y = float(input("Długość drugiego boku:"))
    if invalue_x<0 or invalue_y<0: break

    predicted = model.predict([[invalue_x, invalue_y]])[0][0]
    print(f"Pole prostokąta o bokach {invalue_x:.2f}, {invalue_y:.2f} wynosi \n")
    print(f"model: {predicted:.2f}")
    print(f"rzeczywiście: {invalue_x*invalue_y:.2f}")
    print(f"błąd bezwzględny: {invalue_x*invalue_y-predicted:.2f}")
    print(f"błąd względny: {100*(invalue_x*invalue_y-predicted)/(invalue_x*invalue_y):.2f}%")

