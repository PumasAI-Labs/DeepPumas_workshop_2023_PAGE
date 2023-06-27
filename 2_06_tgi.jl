using DeepPumas, Pumas, CSV, DataFrames, StableRNGs, Random, CairoMakie, PumasPlots, DataFramesMeta

filepath = "data/tgi_os_data.csv"
df = DataFrame(CSV.File(filepath))

ts_pop = read_pumas(
    df,
    observations = [:SLD],
    covariates = [:WT, :AGE, :SEX, :ECOG, :ALBB, :c1, :c2, :c3, :c4, :c5, :c6],
    event_data = false,
)

# Shuffle before splitting
cutoff = round(Int, 0.75 * length(ts_pop))
ts_tpop = ts_pop[1:cutoff]
ts_vpop = ts_pop[cutoff+1:end]

plotgrid(ts_pop[1:20]; xtransform=x->x/365)

# Tumor size model

ts_model = @model begin
  @param begin
    NN_ts ∈ MLPDomain(
      4, 8, 8, (1, softplus);
      reg = L2(0.1; input = false, output = false),
    )
    Ω_ts ~ Constrained(MvNormal(0.1 * I(3)), lower = 0.0)
    ω_m ∈ RealDomain(lower = 0.0)
    sigma ∈ RealDomain(lower = 0.0)
  end
  @random begin
    η_ts ~ MvNormal(Diagonal(Ω_ts))
    log_η_m ~ Normal(0.0, ω_m)
  end
  @derived begin
    iNN_ts := fix(NN_ts, η_ts)
    TS := @. first(iNN_ts(t / 1000))
    η_m := exp(log_η_m)
    SLD ~ @. Normal(η_m * TS, sigma)
  end
end

nsubj = 60
ts_fpm_foce = fit(
  ts_model,
  ts_tpop[1:nsubj],
  sample_params(ts_model),
  MAP(FOCE());
  optim_options = (; iterations=200, show_every=1),
)

ts_pr_foce = predict(ts_fpm_foce)
plotgrid(ts_pr_foce[1:nsubj]; figure=(; resolution=(2000,1200)), linky = false, linkx = false)

ts_sims_foce = [simobs(ts_fpm_foce) for _ in 1:100]
ts_vpc_res_foce = vpc(ts_sims_foce)
vpc_plot(ts_vpc_res_foce, figure = (; resolution = (2500, 1500)), observations = false)

ts_covs = (:c1, :c2, :c3, :c4, :c5, :c6)
# ts_covs = (:WT, :AGE, :ECOG, :ALBB)
ts_target = preprocess(ts_fpm_foce, covs = ts_covs)
ts_target_size = size(ts_target)
ts_nn = MLPDomain(
  ts_target_size[1], 10, 10, 10,
  (ts_target_size[2], identity),
  act = DeepPumas.SimpleChains.softplus,
  reg = L2(0.1),
)
ts_fnn = hyperopt(ts_nn, ts_target)

ts_fnn(ts_pop[1])
ts_fnn(ts_pop[2])

ts_aug_fpm = augment(ts_fpm_foce, ts_fnn, :ts_cov_nn)
ts_aug_coef = coef(ts_aug_fpm)

ts_aug_fpm = fit(
  ts_aug_fpm.model,
  ts_tpop[1:nsubj],
  ts_aug_coef,
  MAP(FOCE());
  optim_options = (; iterations=100, show_every=1),
  checkidentification = false,
  constantcoef = (; NN_ts = ts_aug_coef.NN_ts, ts_cov_nn = ts_aug_coef.ts_cov_nn)
)

ts_aug_coef = coef(ts_aug_fpm)
ts_aug_pr_foce = predict(ts_aug_fpm)
plotgrid(ts_aug_pr_foce[1:nsubj]; figure=(;resolution=(2000,1200)), linky = false, linkx = false)

function pred_loglike(model, subject, param)
  zr = zero_randeffs(model, [subject], param)[1]
  return -conditional_nll(model, subject, param, zr)
end
function pred_loglike(fpm)
  return sum(s -> pred_loglike(fpm.model, s, coef(fpm)), fpm.data)
end

get_pred_accuracy(ts_fpm_foce)
get_pred_accuracy(ts_aug_fpm)
