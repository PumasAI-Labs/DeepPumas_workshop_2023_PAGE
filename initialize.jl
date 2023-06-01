using Pkg

path = dirname(@__FILE__())
Pkg.activate(path)
Pkg.instantiate()
Pkg.precompile()

using DeepPumas
using CairoMakie
x = collect(1:3)
lines(x,x)

## Add some convenience methods that did not make it to the DeepPumas release we'll be using.

(f::DeepPumas.HoldoutValidationResult)(x) = collect(f.ml.post(f.ml.ml(f.ml.pre(x))))
(f::DeepPumas.HyperoptResult)(x) = collect(f.ml.post(f.ml.ml(f.ml.pre(x))))
Pumas.coef(f::DeepPumas.ModelParam) = collect(f.param)
Pumas.coef(f::DeepPumas.HoldoutValidationResult) = coef(f.ml.ml)
Pumas.coef(ho::DeepPumas.HyperoptResult) = coef(ho.ml.ml)

(f::DeepPumas.SimpleChainDomain)(x) = DeepPumas.unalias(f.init(x))
(f::DeepPumas.SimpleChainDomain)(x, p) = DeepPumas.unalias(f.model(x, p))



"""
    preprocess(X, Y; standardize = false)

process the X and Y matrices into a `FitTarget` for use with `fit` or `hyperopt`.

The number of columns in your matrices reflect the number of datapoints while the number of
rows give the dimensionality of your input (X) and output (Y). 

`standardize` toggles a z-score standardization of X. This transformation is stored and can
be applied to new input data fed to a model fitted towards this generated `FitTarget`.
"""
function DeepPumas.preprocess(X::Matrix, Y::Matrix; standardize = false)
  size(X, 2) == size(Y, 2) || throw(
    DimensionMismatch("The number of columns in X and Y does not match in preprocess."),
  )
  xtrsf = if standardize
    meanX = mean(X, dims = 2)
    stdX = std(X, dims = 2)
    zero_inds = findall(==(0), stdX)
    if length(zero_inds) > 0
      @warn "The covariates with indices $zero_inds have 0 standard deviation. This means that they can be removed from the fitting since they should not contribute to the prediction of the neural network."
      stdX[zero_inds] .= 1
    end
    X = (X .- meanX) ./ stdX
    x -> (x .- meanX) ./ stdX
  else
    identity
  end
  return DeepPumas.FitTarget{:x, :y}(X, Y, xtrsf, identity)
end
