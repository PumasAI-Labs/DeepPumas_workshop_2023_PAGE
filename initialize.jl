using Pkg

path = dirname(@__FILE__())
Pkg.activate(path)
Pkg.instantiate()
Pkg.precompile()

using DeepPumas
using CairoMakie
using StableRNGs
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



# Trigger some time-consuming compilation
let x=1
  datamodel = @model begin
      @param begin
          tvKa ∈ RealDomain()
          tvCL ∈ RealDomain()
          tvVc ∈ RealDomain()
          tvSmax ∈ RealDomain()
          tvn ∈ RealDomain()
          tvSC50 ∈ RealDomain()
          tvKout ∈ RealDomain()
          tvKin ∈ RealDomain()
          Ω ∈ PDiagDomain(5)
          σ ∈ RealDomain()
      end
      @random begin
          η ~ MvNormal(Ω)
      end
      @pre begin
          Smax = tvSmax * exp(η[1]) 
          SC50 = tvSC50 * exp(η[2])
          Ka = tvKa * exp(η[3])
          Vc = tvVc * exp(η[4])
          Kout = tvKout * exp(η[5])
          Kin = tvKin
          CL = tvCL
          n = tvn
      end
      @init begin
          R = Kin / Kout
      end
      @vars begin
          cp = max(Central / Vc, 0.)
          EFF = Smax * cp^n / (SC50^n + cp^n)
      end
      @dynamics begin
          Depot' = -Ka * Depot
          Central' = Ka * Depot - (CL / Vc) * Central
          R' = Kin * (1 + EFF) - Kout * R
      end
      @derived begin
          Outcome ~ @. Normal(R, σ)
      end
  end

  p_data = (;
      tvKa = 0.5,
      tvCL = 1.,
      tvVc = 1., 
      tvSmax = 2.9,
      tvn = 1.5,
      tvSC50 = 0.05,
      tvKout = 2.2,
      tvKin = 0.8,
      Ω = Diagonal(fill(0.1, 5)),
      σ = 0.1
  )

  dr = DosageRegimen(1., ii=6, addl=2)
  obstimes = 0:24
  trainpop = synthetic_data(datamodel, dr, p_data; nsubj = 10, obstimes, rng=StableRNG(1))

  model = @model begin
      @param begin
          NN ∈ MLP(5, 6, 5, (1, identity); reg = L2(1.0))
          tvKa ∈ RealDomain(; lower = 0)
          tvCL ∈ RealDomain(; lower = 0)
          tvVc ∈ RealDomain(; lower = 0)
          tvR₀  ∈ RealDomain(; lower = 0)
          ωR₀  ∈ RealDomain(; lower = 0)
          Ω ∈ PDiagDomain(2)
          σ ∈ RealDomain(; lower = 0)
      end
      @random begin
          η ~ MvNormal(Ω)
          η_nn ~ MvNormal(1e-2 .* I(3))
      end
      @pre begin
          Ka = tvKa * exp(η[1])
          Vc = tvVc * exp(η[2])
          CL = tvCL
          R₀ = tvR₀ * exp(10 * ωR₀ * η_nn[1])
          iNN = fix(NN, η_nn)
      end
      @init begin
          R = R₀
      end
      @dynamics begin
          Depot' = -Ka * Depot
          Central' = Ka * Depot - (CL / Vc) * Central
          R' = iNN(Central/Vc, R)[1]
      end
      @derived begin
          Outcome ~ @. Normal(R, σ)
      end
  end

  fpm = fit(
      model,
      trainpop,
      init_params(model),
      MAP(FOCE());
      optim_options = (; iterations=1),
  )
end
