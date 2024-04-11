import numpy as np
import torch
import utils


class Preprocessor:

    """Preprocesses data for the model. Splits the sample into train and test using the dt parameter."""

    def __init__(self, vector_list, sub, dt, coef, train_size=0.7):
        self.vector_list = vector_list
        self.sub = sub
        self.dt = dt
        self.coef = coef
        self.train_size = train_size

        self.nu = 25  # частота видео
        self.mu = 641. / 390.  # частота снимков фМРТ
        self.d1 = self.sub.tensor.shape[0]  # размерности снимка фМРТ до сжатия
        self.d2 = self.sub.tensor.shape[1]
        self.d3 = self.sub.tensor.shape[2]
        self.d4 = self.sub.tensor.shape[3]
        self.d = 2048  # длина вектора признакового описания изображения
        self.N = 641 - int(self.mu * self.dt)  # N - количество снимков фМРТ

        self.train, self.test = self.get_train_test()

    def get_train_test(self):
        pairs = [(int(i * self.nu / self.mu), int(self.mu * self.dt + i))
                 for i in range(self.N)]  # (номер изображения, номер снимка)

        if (self.coef > 1):  # сжатие снимка фМРТ
            #maxpool = torch.nn.MaxPool3d(
            #    kernel_size=self.coef, stride=self.coef)
            maxpool = torch.nn.AvgPool3d(
                kernel_size=self.coef, stride=self.coef)
            input_tensor = self.sub.tensor.permute(3, 0, 1, 2)
            output_tensor = maxpool(input_tensor).permute(1, 2, 3, 0)
            self.sub._tensor = output_tensor
        else:
            self.sub._tensor = self.sub.tensor

        self._d1 = self.sub._tensor.shape[0]
        self._d2 = self.sub._tensor.shape[1]
        self._d3 = self.sub._tensor.shape[2]
        self._d4 = self.sub._tensor.shape[3]

        scans_list = [self.sub._tensor[:, :, :, i]
                      for i in range(self.d4)]  # список тензоров снимков фМРТ
        # список снимков фМРТ, развернутых в векторы
        voxels = [scan.reshape(self._d1 * self._d2 * self._d3).numpy()
                  for scan in scans_list]
        data = [(self.vector_list[n], voxels[k])
                for n, k in pairs]  # (изображение, снимок)

        # train, test
        l = int(self.train_size * self.d4)  # размер обучающей выборки
        train, test = data[:l], data[l:]

        train = [(pair[0], utils.preprocess(pair[1])) for pair in train]
        test = [(pair[0], utils.preprocess(pair[1])) for pair in test]

        return train, test


class LinearModel(Preprocessor):

    """Model that predict next fMRI scan."""

    def __init__(self, vector_list, sub, dt, coef, alpha, train_size=0.7):
        super().__init__(vector_list, sub, dt, coef, train_size)
        self.delta = False
        self.alpha = alpha
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.get_XY()

    def get_XY(self):
        X_train = np.array([pair[0] for pair in self.train])
        Y_train = np.array([pair[1] for pair in self.train]).T
        X_test = np.array([pair[0] for pair in self.test])
        Y_test = np.array([pair[1] for pair in self.test]).T
        return X_train, Y_train, X_test, Y_test

    def fit(self):
        W = []  # матрица весов модели

        if (self.alpha > 0):
            A = np.linalg.inv(self.X_train.T @ self.X_train + self.alpha *
                              np.identity(self.X_train.shape[1])) @ self.X_train.T
        else:
            A = np.linalg.pinv(self.X_train)

        for i in range(self._d1 * self._d2 * self._d3):
            Y_train_vector = self.Y_train[i]
            w = A @ Y_train_vector
            W.append(w)

        self.W = np.array(W)  # w будут строками

    def predict(self):
        self.Y_train_predicted = self.W @ self.X_train.T
        self.Y_test_predicted = self.W @ self.X_test.T

    def evaluate(self):
        self.MSE_train = utils.MSE(self.Y_train_predicted - self.Y_train)
        self.MSE_test = utils.MSE(self.Y_test_predicted - self.Y_test)

class LinearDeltaModel(Preprocessor):

    """Model that predict the difference between current and next fMRI scans."""

    def __init__(self, vector_list, sub, dt, coef, alpha, train_size=0.7):
        super().__init__(vector_list, sub, dt, coef, train_size)
        self.delta = True
        self.alpha = alpha
        self.X_train, self.Y_train, self.deltaY_train, self.X_test, self.Y_test, self.deltaY_test = self.get_XY()

    def get_XY(self):
        delta_train = [(self.train[n][0], self.train[n][1] -
                        self.train[n-1][1]) for n in range(1, len(self.train))]
        delta_test = [(self.test[n][0], self.test[n][1] - self.test[n-1][1])
                      for n in range(1, len(self.test))]
        Y_train = np.array([pair[1] for pair in self.train]).T
        Y_test = np.array([pair[1] for pair in self.test]).T
        X_train = np.array([pair[0] for pair in delta_train])
        deltaY_train = np.array([pair[1] for pair in delta_train]).T
        X_test = np.array([pair[0] for pair in delta_test])
        deltaY_test = np.array([pair[1] for pair in delta_test]).T
        return X_train, Y_train, deltaY_train, X_test, Y_test, deltaY_test

    def fit(self):
        W = []  # матрица весов модели

        if (self.alpha > 0):
            A = np.linalg.inv(self.X_train.T @ self.X_train + self.alpha *
                              np.identity(self.X_train.shape[1])) @ self.X_train.T
        else:
            A = np.linalg.pinv(self.X_train)

        for i in range(self._d1 * self._d2 * self._d3):
            deltaY_train_vector = self.deltaY_train[i]
            w = A @ deltaY_train_vector
            W.append(w)

        self.W = np.array(W)  # w будут строками

    def predict(self):
        self.deltaY_train_predicted = self.W @ self.X_train.T
        self.deltaY_test_predicted = self.W @ self.X_test.T
        self.Y_train_predicted = np.delete(
            self.Y_train, -1, 1) + self.deltaY_train_predicted
        self.Y_test_predicted = np.delete(
            self.Y_test, -1, 1) + self.deltaY_test_predicted

    def evaluate(self, Y_test_predicted=None):
        if Y_test_predicted is None:
            self.MSE_train = utils.MSE(self.Y_train_predicted - np.delete(self.Y_train, 0, 1))
            self.MSE_test = utils.MSE(self.Y_test_predicted - np.delete(self.Y_test, 0, 1))
        else:
            return utils.MSE(Y_test_predicted - np.delete(self.Y_test, 0, 1))

    def repredict(self, W):
        deltaY_test_predicted = W @ self.X_test.T
        Y_test_predicted = np.delete(
            self.Y_test, -1, 1) + deltaY_test_predicted
        return Y_test_predicted
