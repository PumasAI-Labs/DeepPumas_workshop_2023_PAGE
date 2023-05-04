"""
10:15-10:45 DeepPumas IDR model – Automatic identification of dynamics

## Exercise goals
- Run through a DeepPumas workflow to get a feel for it.

Note, we are not providing much context and detail here. We'll pick the individual workflow components apart after having run through it once.

"""

using DeepPumas
using StableRNGs
# using PumasUtilities
using CairoMakie
using Serialization

############################################################################################
## Generate synthetic data from an indirect response model (IDR) with complicated covariates
############################################################################################

## Define the data-generating model
datamodel = @model begin
    @param begin
        tvKa ∈ RealDomain(; lower = 0, init = 0.5)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        tvSmax ∈ RealDomain(; lower = 0, init = 0.9)
        tvn ∈ RealDomain(; lower = 0, init = 1.5)
        tvSC50 ∈ RealDomain(; lower = 0, init = 0.2)
        tvKout ∈ RealDomain(; lower = 0, init = 1.2)
        Ω ∈ PDiagDomain(; init = fill(0.05, 5))
        σ ∈ RealDomain(; lower = 0, init = 5e-2)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @covariates R_eq c1 c2 c3 c4 c5 c6
    @pre begin
        Smax = tvSmax * exp(η[1]) + 3 * c1 / (10.0 + c1) # exp(η[3] + exp(c3) / (1 + exp(c3)) + 0.05 * c4)
        SC50 = tvSC50 * exp(η[2] + 0.5 * (c2 / 20)^0.75)
        Ka = tvKa * exp(η[3] + 0.3 * c3 * c4)
        Vc = tvVc * exp(η[4] + 0.3 * c3)
        Kout = tvKout * exp(η[5] + 0.3 * c5 / (c6 + c5))
        Kin = R_eq * Kout
        CL = tvCL
        n = tvn
    end
    @init begin
        R = Kin / Kout
    end
    @vars begin
        cp = max(Central / Vc, 0)
        EFF = Smax * cp^n / (SC50^n + cp^n)
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = Kin * (1 + EFF) - Kout * R
    end
    @derived begin
        yPD ~ @. Normal(R, σ)
    end
end

## Generate synthetic data.
pop = synthetic_data(
    datamodel;
    covariates = (;
        R_eq = Gamma(5e2, 1/(5e2)), 
        c1 = Gamma(10, 1),
        c2 = Gamma(21, 1),
        c3 = Normal(),
        c4 = Normal(),
        c5 = Gamma(11, 1),
        c6 = Gamma(11, 1),
    ),
    nsubj = 1020,
    rng = StableRNG(123),
)

## Split the data into different training/test populations
trainpop_small = pop[1:80]
trainpop_large = pop[1:1000]
testpop = pop[length(trainpop_large)+1:end]

## Visualize the synthetic data and the predictions of the data-generating model.
pred_datamodel = predict(datamodel, testpop, init_params(datamodel); obstimes = 0:0.1:10);
plotgrid(pred_datamodel)


############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely deterimined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters

model = @model begin
    @param begin
        # Define a multi-layer perceptron (a neural network) which maps from 6 inputs (2
        # state variables + 4 individual parameters) to a single output. Apply L2
        # regularization (equivalent to a Normal prior).
        NN ∈ MLP(6, 6, 5, (1, identity); reg = L2(1.0))
        tvKa ∈ RealDomain(; lower = 0)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower = 0)
    end
    @covariates R_eq
    @random begin
        η ~ MvNormal(Ω)
        η_nn ~ MvNormal(1e-2 .* I(3))
    end
    @pre begin
        Ka = tvKa * exp(η[1])
        Vc = tvVc * exp(η[2])
        CL = tvCL
        R₀ = R_eq

        # Fix individual parameters as static inputs to the NN and return an "individual"
        # neural network:
        iNN = fix(NN, R₀, η_nn)
    end
    @init begin
        R = R₀
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        R' = iNN(Central, R)[1]
    end
    @derived begin
        yPD ~ @. Normal(R, σ)
    end
end

fpm = fit(
    model,
    trainpop_small,
    init_params(model),
    MAP(FOCE());
    # Some extra options to speed up the demo at the expense of a little accuracy:
    optim_options = (; iterations=75),
)


# In case we don't want to wait - I've got a finished fit stored
serialize(@__DIR__() * "/assets/05_fpm.jls", fpm)
# fpm = deserialize(@__DIR__() * "/assets/deep_pumas_fpm.jls")

# The model has succeeded in discovering the dynamical model if the individual predictions
# match the observations well.
pred = predict(model, testpop, coef(fpm); obstimes = 0:0.1:10);
plotgrid(pred)

############################################################################################
## 'Augment' the model to predict heterogeneity from data
############################################################################################
# All patient heterogeneity of our recent model was captured by random effects and can thus
# not be predicted by the model. Here, we 'augment' that model with ML that's trained to 
# capture this heterogeneity from data.
#
# Data quantity is more important for covariate identification than it is for system
# identification. Prediction improvements could still be made with only 80 patients, but
# to correctly identify the covariate effects we need more data.

# Generate a target for the ML fitting from a Normal approximation of the posterior η
# distribution.
target = preprocess(model, trainpop_large, coef(fpm), FOCE())
nn = MLP(numinputs(target), 10, 10, (numoutputs(target), identity); reg = L2())
ho = hyperopt(nn, target)
augmented_model = augment(fpm, ho)

pred_augment =
    predict(augmented_model, testpop, init_params(augmented_model); obstimes = 0:0.1:10);
plotgrid(
    pred_datamodel;
    ipred = false,
    pred = (; color = (:black, 0.4), label = "Best possible pred"),
)
plotgrid!(pred_augment; ipred=false)


############################################################################################
## Further refinement by fitting everything in concert
############################################################################################


# Running this fully would take hours, but we can show that it works
fpm_deep = fit(
  augmented_model,
  trainpop_large,
  init_params(augmented_model),
  MAP(FOCE());
  optim_options = (; time_limit = 5*60),
  diffeq_options = (; alg = Rodas5P()),
  constantcoef = (; NN = init_params(augmented_model).NN)
)
# serialize(@__DIR__() * "/assets/deep_pumas_fpm_deep.jls", fpm_deep)
# fpm_deep = deserialize(@__DIR__() * "/assets/deep_pumas_fpm_deep.jls")

pred_deep = predict(augmented_model, testpop, coef(fpm_deep); obstimes = 0:0.1:10);
plotgrid(
    pred_datamodel;
    ipred = false,
    pred = (; color = (:black, 0.4), label = "Best possible pred"),
)
plotgrid!(pred_deep; ipred=false)
