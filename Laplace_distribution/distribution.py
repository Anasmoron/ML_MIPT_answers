import numpy as np
'''
Мы должны реализовать класс, который:
    1)принимает список значений объектов
    2)оценивает параметры распределения Лапласа 
    3)дает плотность вероятности любого заданного значения объекта.
'''
class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Аргументы:
        -х- массив нумпай формы (n_objects, n_features), в котором
        содержится информация о образцах num_train, каждый размера D
        в данном задании эта функция не заполняется

        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        ####

    def __init__(self, features):
        '''
        Аргументы:
        -features - нумпай массив формы (n_objects, n_features),
        причем в каждой колонке содержатся все возможные значения
        выбранных признаков
        Возможна более простая реализация, но у меня через условие
        проверяется размерность массива
        '''
        ####
        # Do not change the class outside of this block
        if len(features.shape) == 2:#2d случай
            self.loc = np.array([np.median(features[:, 0]), np.median(features[:, 1])])
            self.scale = np.array([(1 / features.shape[0]) * (np.sum(np.abs(features[:, 0] - self.loc[0]))),
                                   (1 / features.shape[0]) * (np.sum(np.abs(features[:, 1] - self.loc[1])))])
        else:#1d случай
            self.loc = np.median(features)  # YOUR CODE HERE
            self.scale = (1 / features.shape[0]) * (np.sum(np.abs(features - self.loc)))
        ####


    def logpdf(self, values):
        '''
        Возвращает логарифм плотности вероятности в каждом введенном значении
        Аргументы:
        -values: нумпаевский массив формы (n_objects,n_features).
        Каждый столбец представляет все возможные значения выбранного признака
        Рассчитано по формуле лапласовского распределения

        '''
        ####
        # Do not change the class outside of this block
        return (-np.log(2. * self.scale)-(np.abs(values-self.loc))/self.scale)
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
