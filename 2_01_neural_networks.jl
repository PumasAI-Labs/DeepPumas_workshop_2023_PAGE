"""
09:20-10:00 Fitting, overfitting and regularizing Neural Networks

## Exercise goal:
- Get a sense of how neural networks can be used to fit data
- See how we risk overfitting
- Learn to mitigate overfitting
- Understand how to balance factors for optimal generalization performance
  - Data availability
  - Data noise
  - Data signal
  - Model size
  - Parameter regularization
"""

using DeepPumas
using CairoMakie

## some API tweaks to enable the example below
macro preprocess(_x, _y)
  :(DeepPumas.FitTarget{$(QuoteNode(_x)), $(QuoteNode(_y))}($(_x), $(_y), identity, identity))
end
(f::DeepPumas.HoldoutValidationResult)(args...) = f.ml.ml(args...)
(f::DeepPumas.HyperoptResult)(args...) = f.ml.ml(args...)
(f::DeepPumas.SimpleChainDomain)(args...) = f.init(args...)
(f::DeepPumas.FluxDomain)(args...) = f.init(args...)

## Generate some data
n = 50
x = rand(Uniform(-1, 1), 1, n)
f(x; σ=0.2) = x .^ 2 + σ * randn()
y = f.(x)


plt = lines(-1:0.01:1, f.(-1:0.01:1; σ=0); label="truth")
scatter!(vec(x), vec(y); label="Data"); plt

## Defined a target x → y mapping and a neural network to fit that mapping.
target = @preprocess x y
nn = MLP(numinputs(target), 5, 5, (1, identity); reg=L1(0.1))

## Fit the neural network
fitted_nn = fit(nn, target; optim_alg=DeepPumas.BFGS());

ŷ = fitted_nn(x)
scatter!(vec(x), vec(ŷ); label="Fitted NN"); plt

## Not too bad!
## But how did we come up with the `L1(0.1)` regularization, and why that NN size?
## Let's see what happens if we remove the regularisation and make the NN bigger.

nn2 = MLP(numinputs(target), 10, 10, (1, identity))
fitted_nn2 = fit(nn2, target; optim_alg=DeepPumas.BFGS());

ŷ = fitted_nn2(x)
scatter!(vec(x), vec(ŷ); label="Over-fitted NN"); plt

## Fit with many different hyperparameters to optimize for generalization performance
nn_ho = hyperopt(nn2, target)
scatter!(vec(x), vec(nn_ho(x)); label="Hyperopt NN"); plt

axislegend(plt.axis; position=:rb); plt



