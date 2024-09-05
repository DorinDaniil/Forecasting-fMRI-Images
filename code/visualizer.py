import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
import seaborn as sns


class Visualizer:

    """Visualizes model data."""

    def __init__(self, model):
        self.model = model
        self.figures = "figures_delta" if self.model.delta else "figures"
        self.filename = f"sub-{self.model.sub.number}-{self.model.dt}-{self.model.coef}"
        if self.model.alpha > 0:
            self.filename += f"-{self.model.alpha}"

    def show_scan_slice(self, scan, scan_number: int, dim: int, slice: int, title, filename_end, mask=None):
        if mask is None:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
                slices = f"-{slice}-_-_"
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
                slices = f"-_-{slice}-_"
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
                slices = f"-_-_-{slice}"
        else:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
                scan_slice_masked = mask[slice, :, :].T
                slices = f"-{slice}-_-_"
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
                scan_slice_masked = mask[:, slice, :].T
                slices = f"-_-{slice}-_"
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
                scan_slice_masked = mask[:, :, slice].T
                slices = f"-_-_-{slice}"
        slice_filename = self.filename + \
            f"-{scan_number}" + slices + filename_end
        self.last_slices = slices
        self.last_slice_filename = slice_filename
        folder_path = os.path.join(os.path.dirname(
            os.getcwd()), self.figures, self.filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # os.chmod(folder_path, 0o400)
        print(title)

        if filename_end == "-difference.png" or filename_end == "-recovered-difference.png" or filename_end == "-delta.png" or filename_end == "-recovered-delta.png":
            plt.imshow(scan_slice, cmap="gray", origin="lower")
        else:
            plt.imshow(scan_slice, cmap="gray", origin="lower", vmin=0, vmax=1)

        plt.colorbar()

        if mask is not None:
            cmap = colors.ListedColormap(['black', 'red'])
            plt.imshow(scan_slice_masked, cmap=cmap, origin="lower", alpha=0.3)
        
        plt.savefig(
            os.path.join(os.path.dirname(os.getcwd()),
                         self.figures, self.filename, slice_filename),
            dpi=300,
            bbox_inches="tight")
        plt.show()

    def _show_scan_test_slice(self, scan: int, dim: int, slice: int, mask=None):
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(scan_test, scan, dim, slice, "TEST", "-test.png", mask)

    def _show_scan_predicted_slice(self, scan: int, dim: int, slice: int, mask=None):
        scan_predicted = self.model.Y_test_predicted.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(scan_predicted, scan, dim,
                             slice, "PREDICTED", "-predicted.png", mask)

    def _show_scan_difference_slice(self, scan: int, dim: int, slice: int, mask=None):
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        scan_predicted = self.model.Y_test_predicted.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        scan_difference = abs(scan_test - scan_predicted)
        self.show_scan_slice(scan_difference, scan, dim,
                             slice, "DIFFERENCE", "-difference.png", mask)

    def _show_recovered_scan_test_slice(self, scan: int, dim: int, slice: int):
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(scan_test, scan, dim, slice,
                             "TEST", "-recovered-test.png")

    def _show_recovered_scan_predicted_slice(self, scan: int, dim: int, slice: int):
        scan_predicted = (self.model.Y_test.T[0] + np.sum(self.model.deltaY_test_predicted.T[:scan], axis=0)).reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(scan_predicted, scan, dim,
                             slice, "PREDICTED", "-recovered-predicted.png")

    def _show_recovered_scan_delta_slice(self, scan: int, dim: int, slice: int):
        scan_delta = np.sum(self.model.deltaY_test_predicted.T[:scan], axis=0).reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(scan_delta, scan, dim, slice,
                             "DELTA", "-recovered-delta.png")

    def _show_recovered_scan_difference_slice(self, scan: int, dim: int, slice: int):
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        scan_predicted = (self.model.Y_test.T[0] + np.sum(self.model.deltaY_test_predicted.T[:scan], axis=0)).reshape(
            (self.model._d1, self.model._d2, self.model._d3))
        scan_difference = abs(scan_test - scan_predicted)
        self.show_scan_slice(scan_difference, scan, dim,
                             slice, "DIFFERENCE", "-recovered-difference.png")

    def show_scan_slices(self, scan: int, dim: int, slice: int, mask=None):
        self._show_scan_test_slice(scan, dim, slice, mask)
        self._show_scan_predicted_slice(scan, dim, slice, mask)
        self._show_scan_difference_slice(scan, dim, slice, mask)

    def show_recovered_scan_slices(self, scan: int, dim: int, slice: int):
        if self.model.delta == False:
            raise ValueError(
                "Данный метод доступен только для LinearDeltaPredictor()")
        self._show_recovered_scan_test_slice(scan, dim, slice)
        self._show_recovered_scan_predicted_slice(scan, dim, slice)
        self._show_recovered_scan_delta_slice(scan, dim, slice)
        self._show_recovered_scan_difference_slice(scan, dim, slice)

    def get_slice_gif(self, dim, slice, filename_end):
        frames = []
        length = self.model.Y_test.shape[1] if self.model.delta == False else self.model.deltaY_test.shape[1]
        for scan in range(length):
            if filename_end == "-test.gif":
                self._show_scan_test_slice(scan, dim, slice)
            elif filename_end == "-predicted.gif":
                self._show_scan_predicted_slice(scan, dim, slice)
            elif filename_end == "-recovered-test.gif":
                self._show_recovered_scan_test_slice(scan, dim, slice)
            elif filename_end == "-recovered-predicted.gif":
                self._show_recovered_scan_predicted_slice(scan, dim, slice)
            # Открываем изображение каждого кадра
            frame = Image.open(os.path.join(os.path.dirname(
                os.getcwd()), self.figures, self.filename, self.last_slice_filename))
            # Добавляем кадр в список с кадрами
            frames.append(frame)
        frames[0].save(
            os.path.join(os.path.dirname(os.getcwd()), self.figures, self.filename,
                         "GIF-" + self.filename + self.last_slices + filename_end),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0
        )

    def get_test_slice_gif(self, dim: int, slice: int):
        self.get_slice_gif(dim, slice, "-test.gif")

    def get_predicted_slice_gif(self, dim: int, slice: int):
        self.get_slice_gif(dim, slice, "-predicted.gif")

    def get_recovered_test_slice_gif(self, dim: int, slice: int):
        self.get_slice_gif(dim, slice, "-recovered-test.gif")

    def get_recovered_predicted_slice_gif(self, dim: int, slice: int):
        self.get_slice_gif(dim, slice, "-recovered-predicted.gif")

    # Распределение значений компонент вектора весов для фиксированного вокселя
    def show_voxel_weight_distribution(self, voxel: int):
        ax = sns.histplot(
            self.model.W[voxel, :], element="poly", linewidth=0, kde=True, bins=20)
        ax.set(xlabel="Значение компоненты вектора весов")
        ax.set(ylabel="Количество компонент")

    # Распределение компонент вектора весов в среднем по всем вокселям
    def show_mean_weight_distribution(self):
        W_mean_rows = np.mean(self.model.W, axis=0)
        ax = sns.histplot(W_mean_rows, element="poly", linewidth=0, kde=True, bins=30)
        #ax.set(xlabel="Значение компоненты вектора весов")
        #ax.set(ylabel="Количество компонент")
        ax.set(xlabel="Weights vector component value")
        ax.set(ylabel="Number of components")
        plt.grid(alpha=0.1)
