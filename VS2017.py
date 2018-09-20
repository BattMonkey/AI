# boston_reg_vsm.py
# regression on the Boston Housing dataset
# Keras 2.1.5 over TensorFlow 1.7.0

import numpy as np
import keras as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg

def my_print(arr, wid, cols, dec):
  fmt = "% " + str(wid) + "." + str(dec) + "f "
  for i in range(len(arr)):
    if i > 0 and i % cols == 0: print("")
    print(fmt % arr[i], end="")
  print("")

def my_accuracy(model, data_x, data_y, pct_close):
  correct = 0; wrong = 0
  n = len(data_x)
  for i in range(n):
    predicted = model.predict(np.array([data_x[i]],
      dtype=np.float32) )  # [[ x ]]
    actual = data_y[i]
    if np.abs(predicted[0][0] - actual) < \
      np.abs(pct_close * actual):
      correct += 1
    else:
      wrong += 1
  return (correct * 100.0) / (correct + wrong)

def main():
  print("\nBegin Boston Houses neural regression demo \n")
  np.random.seed(1)
  kv = K.__version__
  print("Using Keras: ", kv, "\n")

  print("Loading entire 506-item dataset into memory")
  # 506 items min-max, median value / 10
  data_file = ".\\Data\\boston_mm_tab.txt"  
  all_data = np.loadtxt(data_file, delimiter="\t",
    skiprows=0, dtype=np.float32)

  print("Splitting data into 90-10 train-test")
  n = len(all_data)  # number rows
  indices = np.arange(n)  # an array [0, 1, . . n-1]
  np.random.shuffle(indices)     # by ref
  ntr = int(0.90 * n)  # number training items
  data_x = all_data[indices,:-1]  # all rows, skip last col 
  data_y = all_data[indices,-1]  # all rows, just last col
  train_x = data_x[0:ntr,:]  # rows 0 to ntr-1, all cols
  train_y = data_y[0:ntr]    # items 0 to ntr-1 
  test_x = data_x[ntr:n,:]
  test_y = data_y[ntr:n]
  
  print("\nCreating a 13-(10-10)-1 deep neural network")

  my_init = K.initializers.RandomUniform(minval=-0.01,
    maxval=0.01, seed=1)
  simple_sgd = K.optimizers.SGD(lr=0.010) 
  model = K.models.Sequential()
  model.add(K.layers.Dense(units=10, input_dim=13,
    activation='tanh', kernel_initializer=my_init))  # hidden 1
  model.add(K.layers.Dense(units=10,
    activation='tanh', kernel_initializer=my_init))  # hidden 2
  model.add(K.layers.Dense(units=1, activation=None,
    kernel_initializer=my_init))
  model.compile(loss='mean_squared_error',
    optimizer=simple_sgd, metrics=['mse'])

  print("\nStarting training")
  max_epochs = 1000
  h = model.fit(train_x, train_y, batch_size=1,
    epochs=max_epochs, verbose=0)  # use 1 or 2
  print("Training complete")

  acc = my_accuracy(model, train_x, train_y, 0.15) 
  print("\nModel accuracy on train data = %0.2f%%" % acc)

  acc = my_accuracy(model, test_x, test_y, 0.15)
  print("Model accuracy on test data  = %0.2f%%" % acc)

  # mp = ".\\Models\\boston_model.h5"
  # model.save(mp)

  # train_mse = model.evaluate(train_x, train_y, verbose=0)
  # test_mse = model.evaluate(test_x, test_y, verbose=0) 
  # print(train_mse)
  # print(test_mse)

  raw_inpt = np.array([[0.02731, 0.00, 7.070, 0, 0.4690,
    6.4210, 78.90, 4.9671, 2, 242.0, 17.80, 396.90,
    9.14]], dtype=np.float32)  

  norm_inpt = np.array([[0.000236, 0.000000, 0.242302,
    -1.000000, 0.172840, 0.547998, 0.782698, 0.348962,
    0.043478, 0.104962, 0.553191, 1.000000, 0.204470]],
    dtype=np.float32)

  print("\nUsing model to make prediction")
  print("Raw input = ")
  my_print(raw_inpt[0], 10, 4, 5)
  print("\nNormalized input =")
  my_print(norm_inpt[0], 10, 4, 5)

  med_price = model.predict(norm_inpt)
  med_price[0,0] *= 10000
  print("\nPredicted median price = ")
  print("$%0.2f" % med_price[0,0])

  print("\nEnd demo ")

if __name__=="__main__":
  main()
