using DeepPumas, Pumas, CSV, DataFrames, StableRNGs, Random, CairoMakie, PumasPlots, DataFramesMeta

filepath = "data/tgi_os_data.csv"
df = DataFrame(CSV.File(filepath))

ts_os_pop = read_pumas(
    df,
    observations = [:SLD, :Death],
    covariates = [:WT, :AGE, :SEX, :ECOG, :ALBB, :c1, :c2, :c3, :c4, :c5, :c6],
    event_data = false,
)

# Shuffle before splitting
cutoff = round(Int, 0.75 * length(ts_os_pop))
ts_tpop = ts_os_pop[1:cutoff]
ts_vpop = ts_os_pop[cutoff+1:end]

plotgrid(ts_os_pop[1:20]; xtransform=x->x/365)

joint_model = @model begin
  @param begin
    NN_λ ∈ MLPDomain(
      4, 7, 7, (1, softplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    base_λ ∈ RealDomain(; lower=1e-8, init=1e-3)
    NN_ts ∈ MLPDomain(
      4, 7, 7, (1, softplus);
      reg = L2(0.01; input = false, output = false, bias = false),
    )
    Ω ~ Constrained(MvNormal(I(3)), lower = 1e-8)
    ω_m ∈ RealDomain(lower = 1e-8)
    sigma ∈ RealDomain(lower = 1e-8)
  end
  @random begin
    η ~ MvNormal(Ω)
    log_η_m ~ Normal(0.0, ω_m)
  end
  @pre begin
    λf = first ∘ NN_λ
    _η = η
    λ_base = base_λ
  end
  @dynamics begin
    it' = 1.0
    Λ' = λf(it / 2500, _η) + λ_base
  end
  @derived begin
    # tumor size
    iNN_ts := fix(NN_ts, η)
    TS := @. first(iNN_ts(t / 2500))
    η_m := exp(log_η_m)
    SLD ~ @. Normal(η_m * TS, η_m * sigma)

    # overall survival
    _λf := first(λf)
    λ := @. _λf(t / 2500, _η) + λ_base
    Death ~ @. TimeToEvent(λ, Λ)
  end
end

nsubj = 30

fpm_laplace = fit(
  joint_model,
  ts_os_pop[1:nsubj],
  sample_params(joint_model),
  LaplaceI(),
  optim_options = (; iterations=200, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
)

joint_fpm_jointmap = fit(
  joint_model,
  ts_os_pop[1:nsubj],
  sample_params(joint_model),
  JointMAP(
    optim_options = (; iterations=200, show_every=1),
    diffeq_options = (; alg = Rodas5P()),
  ),
)

### EXPERIMENTAL ###
joint_fpm_viem = fit(
  joint_model,
  ts_os_pop[1:nsubj],
  sample_params(joint_model),
  Pumas.VIEM(
    presample = true,
    optim_options = (; iterations=200, show_every=1),
    diffeq_options = (; alg = Rodas5P()),
    use_ebes = false,
    nsamples = 5,
    prior = true,
  );
)

# Post-processing - coming soon!
