
# Code for "Exploring Magnetic Fields in Molecular Clouds through Denoising Diffusion Probabilistic Models" ([link](https://ui.adsabs.harvard.edu/abs/2024arXiv241007032X/abstract))

  

## The trained model files are too large to upload to GitHub. Please use the link provided here to download them. [Harvard Dataverse](https://doi.org/10.7910/DVN/NSFBPR)

  

Note: This repository is still under development and lacks a detailed description, but the code is ready for use. If you encounter any issues, feel free to reach out.

  

Before feeding data to the model for predictions, normalize the input features as follows:

  

1.  ****Density (N):**** Normalize the density (N, in cm⁻²) using the equation:  N_norm = (log₁₀(N) - 19) / 6

  

2.  ****Angle (B_angle):**** Decompose the angle into two components:

* B_angle_pi = B_angle mod π  (using the modulo operator)

* B_angle_pi_2 = B_angle_pi * _2_

* comp_cos = cos(B_angle_pi_2) * _0.5 + 0.5_

* comp_sin = sin(B_angle_pi_2) * _0.5 + 0.5_

  

3.  ****Velocity Dispersion (vel):**** Normalize the velocity dispersion (vel, in cm/s) using the equation: vel_norm = (log₁₀(vel) - 4) / 2.0

  

The input data for the model should be a vector or array in the following order: [N_norm, comp_cos, comp_sin, vel_norm].
