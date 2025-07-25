# Physics Informed Neural Networks for Euler-Bernoulli Bending

- This repository contains the code I developed to train PINNs for the Euler-Bernoulli Beam bending problem as a URA. I used Pytorch and Python.
- The PINN predicts the deflection and its derivatives at 100 points along the beam under a uniformly distributed load.
- The PDE it is trained on is:

$$
EI \frac{d^4 y}{dx^4} = w
$$

- There are two main files:
  - "main_data" is a data-driven PINN model, using both data loss and PDE loss to train the network. Data was generated using the FEA code I developed for Euler-Bernoulli beams: https://github.com/duck-wad/FEA-of-Euler-Bernoulli-Beams
  - "main_nodata" is a purely physics trained PINN model, using only the PDE and BC loss.

<img width="708" height="417" alt="image" src="https://github.com/user-attachments/assets/32151697-13eb-4ca4-be24-93cd30d49347" />

<img width="889" height="1069" alt="image" src="https://github.com/user-attachments/assets/75c0174f-0611-4a40-a5da-6b80db53e8a3" />
