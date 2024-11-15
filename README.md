<div align="center">
<h1>Forecasting fMRI Images From Video Sequences: Linear Model Analysis </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email: *</sup>, [Nikita Kiselev](https://github.com/kisnikser)<sup>1 *</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>

<sup>1</sup> Moscow Institute of Physics and Technology

<sup>:email:</sup> Corresponding author, <sup>*</sup> Equal contribution

[📝 Paper (Health Information Science and Systems 2024)](https://doi.org/10.1007/s13755-024-00315-5), [</> Code](https://github.com/DorinDaniil/Forecasting-fMRI-Images/tree/main/code), [🎬 Video](https://www.youtube.com/live/WnIRaRl730A?si=Txo-uVvyS6JaTzRT&t=4305), [🎫 Poster](https://github.com/DorinDaniil/Forecasting-fMRI-Images/blob/main/poster/poster.pdf)

</div>

## 💡 Abstract
The issue of reconstructing the relationship between functional magnetic resonance imaging (fMRI) sensor readings and human perception of the external is investigated. The study analyzes the dependence between the fMRI images and the videos viewed by individuals. Based on this analysis, a method is proposed for approximating the fMRI readings using the video sequence. The method is based on the assumption that there is a time-invariant hemodynamic response to changes in blood oxygen levels. A linear model is constructed for each individual voxel in the fMRI image, assuming that the image sequence follows a Markov property. To test the proposed method, a computational experiment was conducted on a dataset collected during tomographic examinations of a large number of individuals. The performance of the method was evaluated based on the experimental data, and hypotheses were tested regarding the invariance of the model weights and the correctness of the method.

## 🔎 Overview
<div align="center">
  <img alt="overview" src="https://github.com/DorinDaniil/Forecasting-fMRI-Images/assets/70231416/b02ebddd-432c-4e7b-8c81-905a99ded757">
</div>

## 🛠️ Repository Structure
The repository is structured as follows:
- `paper`: This directory contains the main paper in PDF format (`main.pdf`) and the LaTeX source file (`main.tex`). Also there is a directory `figures` with images used in the paper.
- `code`: This directory contains the code used in the paper. It has its own `README.md` file providing a detailed description of the code files.
```shell
Forecasting-fMRI-Images
├── LICENSE
├── README.md
├── code
│   ├── README.md
│   ├── dataloader.py
│   ├── main.ipynb
│   ├── models.py
│   ├── utils.py
│   └── visualizer.py
└── paper
    ├── figs
    ├── main.pdf
    ├── main.tex
    ├── references.bib
    ├── sn-jnl.cls
    └── sn-mathphys-num.bst
```
