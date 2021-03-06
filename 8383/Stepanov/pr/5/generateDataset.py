import numpy as np

x = np.random.normal(3.0, 10.0, 3000).reshape(-1,1) # генерируем х
e = np.random.normal(0, 0.3, 3000).reshape(-1,1)  # генерируем ошибку

y = np.sin(x/2) + e # считаем y

np.savetxt('train_dataset.csv', np.hstack((x, y))) # записываем в файл

x = np.random.normal(3.0, 10.0, 1000).reshape(-1,1)
e = np.random.normal(0, 0.3, 1000).reshape(-1,1)

y = np.sin(x/2) + e

np.savetxt('test_dataset.csv', np.hstack((x, y)))