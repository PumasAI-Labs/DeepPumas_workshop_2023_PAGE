using DeepPumas, Pumas, CSV, DataFrames, StableRNGs, Random, CairoMakie, PumasPlots, DataFramesMeta
const ssoftplus = DeepPumas.SimpleChains.softplus

filepath = "data/tgi_os_data.csv"
df = DataFrame(CSV.File(filepath))

os_pop = read_pumas(
    df,
    observations = [:Death],
    covariates = [:WT, :AGE, :SEX, :ECOG, :ALBB, :c1, :c2, :c3, :c4, :c5, :c6],
    event_data = false,
)

os_cutoff = round(Int, 0.75 * length(os_pop))
os_tpop = os_pop[1:os_cutoff]
os_vpop = os_pop[os_cutoff+1:end]

os_model1 = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      1, 15, 15, (1, ssoftplus);
      reg = L2(0.01; input = true, output = true, bias = true),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
  end
  @pre begin
    λf = first ∘ NN_λ
    _base_λ = base_λ
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 2000) + _base_λ
  end
  @derived begin
    _λf := first ∘ NN_λ
    λ := @. _λf(t / 2000) + _base_λ
    Death ~ @. TimeToEvent(λ, Λ)
  end
end

os_model2 = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      6, 15, 15, (1, ssoftplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
  end
  @covariates begin
    AGE
    WT
    ECOG
    ALBB
    SEX        
  end
  @pre begin
    λf = first ∘ NN_λ
    _base_λ = base_λ
    covs = (
      AGE / 70,
      WT / 100,
      ECOG / 2,
      ALBB / 200,
      SEX == "Male" ? 0.0 : 1.0,
    )
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 2000, covs) + _base_λ
  end
  @derived begin
    λf1 := λf[1]
    λ := @. λf1(t / 2000, covs) + _base_λ
    Death ~ @. TimeToEvent(λ, Λ)
  end
end

os_model3 = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      7, 15, 15, (1, ssoftplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
  end
  @covariates begin
    c1
    c2
    c3
    c4
    c5
    c6
  end
  @pre begin
    λf = first ∘ NN_λ
    _base_λ = base_λ
    covs = (c1, c2, c3, c4, c5, c6)
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 2000, covs) + _base_λ
  end
  @derived begin
    λf1 := λf[1]
    λ := @. λf1(t / 2000, covs) + _base_λ
    Death ~ @. TimeToEvent(λ, Λ)
  end
end

nsubj = min(length(os_tpop), 150)
_os_tpop = os_tpop[1:nsubj]

fpm1 = fit(
  os_model1,
  _os_tpop,
  sample_params(os_model1),
  MAP(NaivePooled()),
  optim_options = (; iterations=200, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

fpm2 = fit(
  os_model2,
  _os_tpop,
  sample_params(os_model2),
  MAP(NaivePooled()),
  optim_options = (; iterations=200, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

fpm3 = fit(
  os_model3,
  _os_tpop,
  sample_params(os_model3),
  MAP(NaivePooled()),
  optim_options = (; iterations=200, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

# Training log likelihood
loglikelihood(os_model1, _os_tpop, coef(fpm1), NaivePooled())
loglikelihood(os_model2, _os_tpop, coef(fpm2), NaivePooled())
loglikelihood(os_model3, _os_tpop, coef(fpm3), NaivePooled())

# Test log likelihood
loglikelihood(os_model1, os_vpop, coef(fpm1), NaivePooled())
loglikelihood(os_model2, os_vpop, coef(fpm2), NaivePooled())
loglikelihood(os_model3, os_vpop, coef(fpm3), NaivePooled())
