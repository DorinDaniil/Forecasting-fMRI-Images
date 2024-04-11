import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
import nibabel as nib
from tqdm import tqdm


class FramesLoader:

    """Load and prepare vidoe for models."""

    def __init__(self) -> None:
        """Initialize paths for video and frames."""
        self.fig_path = os.path.join(os.path.dirname(os.getcwd()), "figures")
        self.video_path = os.path.join(os.path.dirname(
            os.getcwd()), "src", "Film stimulus.mp4")
        self.videocap = cv2.VideoCapture(self.video_path)
        self.preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms()

    def _video_to_frames(self):
        """Import video from video_path and divide it into the frames."""
        success, frame = self.videocap.read()
        count = 1
        while success:
            cv2.imwrite(os.path.join(os.path.dirname(os.getcwd()),
                        "src", "frames", f"frame_{count}.jpg"), frame)
            success, frame = self.videocap.read()
            count += 1

    def _frames_to_tensors(self):
        """Takes frames from video and transform them into the tensors."""

        #preprocess = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #])

        for i in range(1, 9751):
            frame_path = os.path.join(os.path.dirname(
                os.getcwd()), "src", "frames", f"frame_{i}.jpg")
            frame = Image.open(frame_path)
            frame_tensor = self.preprocess(frame)
            frame_tensor = frame_tensor.unsqueeze(0)
            yield frame_tensor

    def _tensors_to_vectors(self):
        """Takes frames tensors and transform them into the vectors using ResNet152."""

        #model = torch.hub.load('pytorch/vision:v0.6.0',
        #                       'resnet152', pretrained=True, verbose=False)
        #
        #for param in model.parameters():
        #    param.requires_grad = False
        #
        #model.fc = torch.nn.Identity()

        model = torchvision.models.resnet152(weights='IMAGENET1K_V2')

        for frame_tensor in tqdm(self._frames_to_tensors()):
            # передача картинки в модель и получение выходных данных
            with torch.no_grad():
                output = model(frame_tensor)
            # преобразование выходных данных в вектор
            vector = output.numpy().flatten()
            yield vector

    def get_vector_list(self, load=True, new=False):
        """Returns vector_list which consists of frames embeddings."""
        if load == True:
            filename = "vector_list.npy" if new == False else "vector_list_new.npy"
            vector_list = np.load(os.path.join(
                os.path.dirname(os.getcwd()), "src", filename))
        else:
            filename = "vector_list" if new == False else "vector_list_new"
            vector_list = [vector for vector in self._tensors_to_vectors()]
            np.save(filename, vector_list)
        return vector_list

class Sub:

    """Responsible for the subject and contains information about his data."""

    subs_with_fmri = ['04', '07', '08', '09', '11', '13', '14', '15', '16', '18',
                      '22', '24', '27', '28', '29', '31', '35', '41', '43', '44',
                      '45', '46', '47', '51', '52', '53', '55', '56', '60', '62']

    def __init__(self, number):
        if not number in Sub.subs_with_fmri:
            raise ValueError(f"У {number} испытуемого отсутствуют снимки фМРТ")
        else:
            self.number = number
        self.path = os.path.join(os.path.dirname(os.getcwd()), "src", "ds003688-download", f"sub-{self.number}",
                                 "ses-mri3t", "func", f"sub-{self.number}_ses-mri3t_task-film_run-1_bold.nii.gz")
        self.scan = nib.load(self.path)
        self.data = self.scan.get_fdata()
        self.tensor = torch.tensor(self.data)
        self.tensor_np = self.tensor.numpy()
