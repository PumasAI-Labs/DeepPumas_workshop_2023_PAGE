# Scientific Modeling Augmented by Machine-Learning with DeepPumas

DeepPumas is not an incremental improvement but a game-changer in pharmaceutical drug development analytics. 

### About this Event

Scientific Modeling Augmented by Machine-Learning with DeepPumas
Presented by Pumas-AI, Inc.
Often scientists may not know the exact biological mechanisms dictating the data signatures. For example, which risk factors drive responders to a cancer treatment or how a biomarker is related to clinical outcomes. Current data science methods require big data, and they ignore prior knowledge of the problem at hand. Pumas-AI is poised to disrupt this. DeepPumasTM enables seamless integration of domain-specific knowledge and data-science methodology, reducing dependence on data size and enabling faster decision-making.

Here, we will learn, hands-on, how DeepPumas can automatically discover complex predictive factors to individualize predictions. Furthermore, we will learn how dynamical systems that model the longitudinal evolution of patient outcomes can be augmented by machine learning – enabling data-driven discovery of the underlying biology. Together, this enables effective use of data to rapidly develop models that predict individual outcomes from heterogeneous sources of patient data.

Applicable across the whole chain of drug development, from lead generation, quality by design manufacturing, clinical research, and market research to individualized patient management, DeepPumas is not an incremental improvement but a game-changer.

The workshop is split in two, where the first day is dedicated to learning the powerful Pumas software for pharmacometric modeling. During the second day, we learn about machine learning and how it can be seamlessly embedded in pharmacometric models using DeepPumas.

### At the end of the workshop, participants will:

- Gain an overview of the Pumas modelling and simulation ecosystem
- Gain an overview of the theory behind neural networks and their embedding within dynamical and statistical models.
- Learn workflows for incorporating machine learning into scientific models
- Use neural-embedded pharmacological models to identify predictive factors in patient data.
- Use embedded neural networks to identify missing terms and unknown relationships in pharmacological models.
- Gain an overview of what problems can be solved by this novel technology.

### Workflows covered:

- Model population pharmacokinetics in Pumas
- Model PKPD in Pumas
- Model time-to-event in Pumas
- Model mechanistic PKPD data using DeepPumas.
- Automatically identify the equations that drive the drug response.
- Identify complex relationships between covariates and patient outcomes to make personalized baseline predictions.
- Tumor size survival analysis.
- Use DeepPumas to automatically identify tumor size dynamics
- Use DeepPumas to automatically Identify the effect of tumor size on survival.

### June 26, 2023

08:30-08:45 Introduction to Pumas and DeepPumas

08:45-09:15 Setting up the workspace

09:15-10:00 Population Pharmacokinetic modeling Part 1 - Overview of Pumas and Pumas Workflow

10:00-10:15 Coffee break

10:15-11:15 Population Pharmacokinetic modeling Part 2 - Iterative Model Building

11:15-13:00 Sequential PKPD – Indirect response model

13:00-14:00 Lunch break

14:00-15:00 Survival modeling

15:00-16:15 Data Wrangling


### June 27, 2023

08:30-09:20 Machine learning & Neural Networks (Lecture)

09:20-10:00 Fitting, overfitting and regularizing Neural Networks

10:00-10:15 Coffee Break

10:15-10:45 DeepPumas IDR model – Automatic identification of dynamics

10:45-11:30 Why did that work? DeepPumas theory (lecture)

11:30-12:30 DeepPumas IDR model – Identification of prognostic factors

12:30-13:30 Lunch Break

13:30-14:30 DeepPumas Tumor size modeling – identifying dynamics

14:30-15:15 DeepPumas survival modeling – automatic identification of hazard

15:15-15:30 Coffee Break

15:30-16:30 DeepPumas joint tumor size inhibition and overall survival model – automatic identification of tumor size effect on hazard.

16:30-17:00 Concluding remarks and outlining the future.


## Getting started

This workshop will be run in a Pumas Enterprise cloud-based environment. To get started, we'll need to go

- Go to https://pumasai.juliahub.com and log in
- Launch the "Julia IDE" application
- Once launched and you're in a VSCode editor that'll open up, clone this repository
  - open up a terminal (Ctrl+shift+p then `Terminal: Focus on terminal view`).
  - Make sure you're in the `~/data/code` folder
  - run 
```
git clone https://github.com/PumasAI-Labs/DeepPumas_workshop_2023_PAGE.git
```

This will get you all you need, but for the SciTrek trial run, we'll have some additional set up to do becaue we did not have time to create a DeepPumas application where all of this is hidden.

- Run `initialize.jl`. This will install what you need and do some compilation. It should take ~20 minutes so we'll do this before the lectures.

