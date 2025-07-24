# Physics Informed Neural Networks for Euler-Bernoulli Bending

- This repository contains the code I developed to train PINNs for the Euler-Bernoulli Beam bending problem as a URA. I used Pytorch and Python.
- The PINN predicts the deflection and its derivatives at 100 points along the beam under a uniformly distributed load.
- The PDE it is trained on is:

<img width="324" height="171" alt="image" src="https://github.com/user-attachments/assets/1e31fc5b-7ecf-4c70-8a7d-7ccbef879f01" />

- There are two main files:
  - "main_data" is a data-driven PINN model, using both data loss and PDE loss to train the network. Data was generated using the FEA code I developed for Euler-Bernoulli beams: https://github.com/duck-wad/FEA-of-Euler-Bernoulli-Beams
  - "main_nodata" is a purely physics trained PINN model, using only the PDE and BC loss.

<img width="708" height="417" alt="image" src="https://github.com/user-attachments/assets/32151697-13eb-4ca4-be24-93cd30d49347" />

<img width="712" height="855" alt="image" src="https://github.com/user-attachments/assets/7b642b86-850e-4b58-8ad4-70db64a452f4" />
