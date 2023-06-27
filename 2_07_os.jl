using DeepPumas, Pumas, CSV, DataFrames, StableRNGs, Random, CairoMakie, PumasPlots, DataFramesMeta
const ssoftplus = DeepPumas.SimpleChains.softplus

filepath = "data/osdata.csv"
df = DataFrame(CSV.File(filepath))

os_pop = read_pumas(
    df,
    observations = [:DV],
    covariates = [:DOSE],
    event_data = false,
)

os_cutoff = round(Int, 0.75 * length(os_pop))
os_tpop = os_pop[1:os_cutoff]
os_vpop = os_pop[os_cutoff+1:end]

os_model1 = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      1, 10, 10, (1, ssoftplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
  end
  @pre begin
    λf = first ∘ NN_λ
    _base_λ = base_λ
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 500) + _base_λ
  end
  @derived begin
    _λf := first ∘ NN_λ
    λ := @. _λf(t / 500) + _base_λ
    DV ~ @. TimeToEvent(λ, Λ)
  end
end

os_model2 = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      2, 15, 15, (1, ssoftplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
  end
  @covariates DOSE
  @pre begin
    λf = first ∘ NN_λ
    dose = DOSE
    _base_λ = base_λ
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 500, dose) + _base_λ
  end
  @derived begin
    λf1 := λf[1]
    λ := @. λf1(t / 500, dose) + _base_λ
    DV ~ @. TimeToEvent(λ, Λ)
  end
end

fpm1 = fit(
  os_model1,
  os_tpop,
  sample_params(os_model1),
  MAP(NaivePooled()),
  optim_options = (; iterations=500, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

fpm2 = fit(
  os_model2,
  os_tpop,
  sample_params(os_model2),
  MAP(NaivePooled()),
  optim_options = (; iterations=500, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

# Training log likelihood
loglikelihood(os_model1, os_tpop, coef(fpm1), NaivePooled())
loglikelihood(os_model2, os_tpop, coef(fpm2), NaivePooled())

# Test log likelihood
loglikelihood(os_model1, os_vpop, coef(fpm1), NaivePooled())
loglikelihood(os_model2, os_vpop, coef(fpm2), NaivePooled())

vpc1 = vpc(fpm1)
vpc2 = vpc(fpm2)

vpc_plot(vpc1)
vpc_plot(vpc2)
