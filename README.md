# Biomass Pyrolysis in a Fluidized Bed Reactor

This repository presents a multi-scale modeling framework for simulating biomass pyrolysis, combining intra-particle transport and reaction phenomena with a 2D fluidized bed (FB) reactor. The project was developed as part of the **6EMA06 Multiphase Reactor Modelling** course at TU/e.

## Overview

Biomass pyrolysis is a thermochemical process for converting organic materials into tar, gas, and char. Fluidized beds are often used due to their excellent mixing and heat transfer. In this work, a coupled model was developed that:

- Resolves heat and mass transport within a single biomass particle using a progressive conversion scheme (PCM)
- Simulates fluidized bed hydrodynamics using an extended Kunii-Levenspiel model
- Integrates particle- and reactor-level behavior through mass fluxes and source term coupling

## Model Features

### 🔬 Particle-Scale Model
- Spherical particle geometry (1D radial domain)
- Moisture evaporation, primary pyrolysis, and secondary tar cracking
- Heat and species transport via diffusion and convection
- Porosity and thermal conductivity evolving dynamically with conversion
- Solved using finite volume method and backward Euler scheme

### 🌀 Reactor-Scale Model
- 2D cylindrical fluidized bed domain (r–z)
- Three-phase description: bubble, emulsion, and solid (CSTR)
- Mass transfer correlations (KL model, Sherwood/Nusselt numbers)
- Plug flow for bubble/emulsion, dispersion-enhanced mixing for solid phase

### 🔗 Coupling Strategy
- Particle surface fluxes are injected as boundary conditions for the FB reactor
- Reaction rates are volume- and time-averaged to serve as source terms
- Feedback loop allows for simulation of dynamic species evolution across scales

## Results

- Validated particle model against analytical solutions for pure kinetics, convection, and diffusion
- Explored the impact of particle size on intra-particle profiles and volatile yields
- Simulated 2D reactor species fields showing radial gradients due to limited transport near walls
- Identified optimal operating regimes for maximizing tar yield while limiting secondary reactions

<p align="center">
  <img src="images/Particle_DensityR5e-3.png" width="450" alt="Density Evolution in Biomass Particle">
  <img src="images/Gas and Tar results.png" width="450" alt="2D Tar and Gas Profiles in FB Reactor">
</p>

## Repository Contents

- `/report/` – Final project report (PDF)
- `/src/` – Python scripts for the particle and reactor models
- `/figures/` – Generated plots and visualizations
- `README.md` – Project overview

## Authors

- David S. Warrand  
- Oisín B. Rutgers Kearns  
- Rico-Léon W.J. Verhagen  
- **Adam J. Misa**  

MSc Chemical Engineering – TU/e  
📧 aj.misa@outlook.com

## License

This project is licensed for academic and educational use only.
