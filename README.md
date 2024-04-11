# Forecasting fMRI Images From Video Sequences: Linear Model Analysis

**Author:** Daniil Dorin, Nikita Kiselev

**Advisor:** Andrey Grabovoy

## Abstract
The problem of reconstructing the dependence between fMRI sensor readings 
and human perception of the external world is investigated. 
The dependence between the sequence of fMRI images and the video sequence 
viewed by a person is analyzed. Based on the dependence study, a method for 
approximating fMRI readings from the viewed video sequence is proposed. 
The method is constructed under the assumption of the presence of a time 
invariant hemodynamic response time dependence of blood oxygen level. 
A linear model is independently constructed for each voxel of the fMRI image.  
The assumption of markovality of the fMRI image sequence is used. 
To analyze the proposed method, a computational experiment is performed 
on a sample obtained during tomographic examination of a large number of subjects. 
The dependence of the method performance quality on the hemodynamic response 
time is analyzed on the experimental data. Hypotheses about invariance of 
model weights with respect to a person and correctness of the constructed 
method are tested.

## Repository Structure
The repository is structured as follows:

- `paper`: This directory contains the main paper in PDF format (`main.pdf`) and the LaTeX source file (`main.tex`). Also there is a directory `figs` with images used in the paper.
- `code`: This directory contains the code used in the paper. It has its own `README.md` file providing a detailed description of the code files.
