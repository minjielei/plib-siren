# Differentiable Photon Library Representation with Siren

## Background
[SIREN](https://github.com/vsitzmann/siren), a neural scene representation model, is applied to learn a differentiable version of a photon library lookup table. Leveraging periodic activation functions for implicit neural representations, SIREN can accurately represent complex signals and their derivatives. Such a differentiable version of the photon libray lookup table with accurate gradient surface representation can then be used as part of a differentiable simulator chain to optimize and automate the calibration and reconstrucion process for a wide range of particle physics experiments. 

### Photon Library
A lookup table containing the probability value for a photon produced inside a volume to be detected by an optical detector. Due to the large and complex detector geometry of modern particle physics experiments, a photon library lookup table is often used in place of challenging and time-consumming real time ray-tracing. Here, the specific instance of a photon library table for the ICARUS experiment is used, which has a dimension of (74, 77, 394, 180), with the last dimension corresponding to 180 optical detectors called PMTs. 
