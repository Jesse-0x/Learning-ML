import keras
model = keras.models.load_model('my_model.h5')

def f(X):
    return 2*X-1

offpersontage = []
for i in range(-100, 10000):
    comp = f(i) - model.predict([[i]]).tolist()[0][0]
    offpersontage.append(comp)

c = 0
for j in offpersontage:
    c += j

print(f'{c/len(offpersontage):.22f}')