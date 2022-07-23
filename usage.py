import keras
print('Hello, there!')
print("This is a model for given the n'th even number!")
print("It's a simple neural network with one hidden layer.")
print("It's trained on a dataset of 31 numbers.")
print("It's trained over 10000 epochs.")

print("Wish u like it XD")

model = keras.models.load_model('my_model.h5')

while True:
    n = float(input("Enter the number: "))
    print(round(model.predict([[n]]).tolist()[0][0]))
