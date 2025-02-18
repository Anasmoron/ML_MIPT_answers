import numpy as np
from scipy import stats #для predict_labels
class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Тренирует классификатор. Для кНН это просто штучка, запоминающая инфу.
        ВВОД:
        -Х : это нумпаевский массив размера (num_train,D). В нем содержатся данные для обучения .
        Всего num_train образцов , каждый из которых содержит D признаков
        -y : нумпаевский массив формы (N,),  содержающий метки для обучения. y[i] - метка X[i]

        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """ ПРЕДСКАЗЫВАЕТ МЕТКИ ДЛЯ ТЕСТОВЫХ ДАННЫХ, используя этот классификатор
        ВВОД:
        - Х: - тот же нумпаевский массив с данными(num_test) и их признаками (D). Только теперь
        мы берем данные для валидации
        -к: -число ближайших соседий, которые "голосуют" за предсказание метки
        -num_loops: -метод, который будет использоваться для поиска расстояния между двумя точками
        ВОЗВРАТ:
        -у: нумпаевский массив формы (num_test,), содержащий предсказанные метки для тестоых данных.
         y[i]-предсказанная метка для тестового данного X[i]


        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i][j]=np.sqrt(np.sum((X[i]-self.X_train[j])**2))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dist_arr=(self.X_train-X[i])**2
            dists[i]=np.sqrt(np.sum(dist_arr,axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        sum_X_square = np.sum(X ** 2, axis=1).reshape(-1, 1)
        sum_train_square = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
        cross_term = X.dot(self.X_train.T)
        dists_squared = sum_X_square + sum_train_square - 2 * cross_term
        dists = np.sqrt(np.maximum(dists_squared, 0))
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Матрица расстояний предсказывает метку каждого тестового данного
        ВВОД:
        -dists: нумпаевский массив размера (num_test,num_train). Элемент этой матрицы-
        расстояние между i-ой тестовой j-ой обученной точкой

        ВЫВОД:
        -у: нумпаевский массив (num_test,), содержащий предсказанные метки для валидационных данных.
        y[i]-метка для  X[i] валидационного(тестового) данного^^

        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            '''num_in - массив индексов, содержащих минимальное расстояние к-ближайших соседей
               closest_y - массив с метками ближайших к-соседей'''
            num_in=np.argsort(dists[i,:])[:k]
            closest_y = self.y_train[num_in]


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ''' вычисляет самое часто встречающееся значение. 
            содержит два поля - mode (значение самое частое)
            count (число появлений самого частого значения). 
            ИЗ библиотеки scipy'''
            y_pred[i] = stats.mode(closest_y)[0]

        return y_pred
