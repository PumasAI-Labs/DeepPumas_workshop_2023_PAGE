using DeepPumas, Pumas, CSV, DataFrames, StableRNGs, Random, CairoMakie, PumasPlots, DataFramesMeta

filepath = "data/tgi_os_data.csv"
df = DataFrame(CSV.File(filepath))
@rsubset! df :time <= 1000

os_pop = read_pumas(
    df,
    observations = [:Death],
    covariates = [:SLD],
    event_data = false,
)

os_cutoff = round(Int, 0.75 * length(os_pop))
os_tpop = os_pop[1:os_cutoff]
os_vpop = os_pop[os_cutoff+1:end]

os_sim_model = @model begin
  @param begin
   a ∈ RealDomain()
  end
  @covariates SLD
  @pre begin
    λ = 1e-3 * SLD + 1e-4 * SLD^2
  end
  @dynamics begin
    Λ' = λ
  end
  @derived begin
    DV ~ @. TimeToEvent(λ, Λ)
  end
end

sims = simobstte(
  os_sim_model,
  os_pop,
  sample_params(os_sim_model),
  optim_options = (; iterations=500, show_every=1),
  diffeq_options = (; alg = Rodas5P()),
  maxT = 1000.0,
)

CSV.write("tgi_os_data2.csv", DataFrame(sims))
