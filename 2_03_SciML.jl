using DeepPumas
using CairoMakie
using StableRNGs

# 
# TABLE OF CONTENTS
# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 

"""
Helper Pumas model to generate synthetic data. It assumes 
one compartment non-linear elimination and oral dosing.
"""
data_model = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower = 0.0)  # typical value of maximum inhibition
        tvIC50 ∈ RealDomain(; lower = 0.0)  # typical value of concentration for half-way inhibition
        tvKa ∈ RealDomain(; lower = 0.0)    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)       # residual error
    end
    @pre begin
        Imax = tvImax                       # per subject value = typical value,
        IC50 = tvIC50                       # that is, no subject deviations, or,
        Ka = tvKa                           # in other words, no random effects
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

true_parameters = (; tvImax = 1.1, tvIC50 = 0.8, tvKa = 1.0, σ = 0.1)

# 1.1. Simulate subjects A and B with different dosage regimens

data_a = synthetic_data(data_model, DosageRegimen(5.), true_parameters; nsubj=1, obstimes=0:1:10, rng=StableRNG(1))
data_b = synthetic_data(data_model, DosageRegimen(10.), true_parameters; nsubj=1, obstimes=0:1:10, rng=StableRNG(2))

plotgrid(data_a; data = (; label = "Data (subject A)"))
plotgrid!(data_b; data = (; label = "Data (subject B)"), color = :gray)

# 1.2. A dummy neural network for modeling dynamics

time_model = @model begin
    @param begin
        mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L2(0.5))
        σ ∈ RealDomain(; lower = 0.0)
    end
    @pre nn_output = mlp(t)[1]
    @derived Outcome ~ @. Normal(nn_output, σ)
end

# Strip the dose out of the subject since this simple model does not know what to do with a dose.
data_a_no_dose = read_pumas(DataFrame(data_a); observations = [:Outcome], event_data = false)
data_b_no_dose = read_pumas(DataFrame(data_b); observations = [:Outcome], event_data = false)

fpm_time = fit(time_model, data_a_no_dose, init_params(time_model), MAP(NaivePooled()))

pred_a = predict(fpm_time; obstimes=0:0.1:10);
plotgrid!(pred_a; pred = (; label = "Pred (subject A)"), ipred = false)

pred_b = predict(time_model, data_b_no_dose, coef(fpm_time); obstimes=0:0.1:10);
plotgrid!(pred_b, pred = (; label = "Pred (subject B)", color = :red), ipred = false)

# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 2.1. Delegate the identification of dynamics to a neural network

ude_model = @model begin
    @param begin
        mlp ∈ MLPDomain(2, 6, 6, (1, identity); reg = L2(0.5))    # neural network with 2 inputs and 1 output
        tvKa ∈ RealDomain(; lower = 0.0)                    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)                       # residual error
    end
    @pre begin
        mlp_ = only ∘ mlp
        Ka = tvKa
    end
    @dynamics begin
        Depot' = -Ka * Depot                                # known
        Central' = mlp_(Depot, Central)                     # left as function of `Depot` and `Central`
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

plotgrid(data_a; data = (; label = "Data (subject A)"))
plotgrid!(data_b; data = (; label = "Data (subject B)"), color = :gray)

fpm_ude = fit(ude_model, data_a, init_params(ude_model), MAP(NaivePooled()))
pred_a = predict(fpm_ude; obstimes=0:0.1:10);
plotgrid!(pred_a; pred = (; label = "Pred (subject A)"), ipred = false)

pred_b = predict(ude_model, data_b, coef(fpm_ude); obstimes=0:0.1:10);
plotgrid!(pred_b, pred = (; label = "Pred (subject B)", color = :red), ipred = false)

# 2.2. Combine existing domain knowledge and a neural network

ude_model_knowledge = @model begin
    @param begin
        mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg = L2(0.5))    # neural network with 1 inputs and 1 output
        tvKa ∈ RealDomain(; lower = 0.0)                          # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)                             # residual error
    end
    @pre begin
        mlp_ = only ∘ mlp
        Ka = tvKa
    end
    @dynamics begin
        Depot' = -Ka * Depot                                # known
        Central' = Ka * Depot - mlp_(Central)               # knowledge of conservation added
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

fpm_knowledge = fit(
    ude_model_knowledge,
    data_a,
    init_params(ude_model_knowledge),
    MAP(NaivePooled());
)

pred_a = predict(fpm_knowledge; obstimes=0:0.1:10);
plotgrid(
    pred_a; 
    ipred = false,
    data = (; label = "Data (subject a)", color = :gray),
    pred = (; label = "Pred (subject a)"),
)

pred_b = predict(ude_model_knowledge, data_b, coef(fpm_knowledge); obstimes=0:0.1:10);
plotgrid!(
    pred_b; 
    ipred = false,
    data = (; label = "Data (subject b)", color = :black),
    pred = (; label = "Pred (subject b)", color=:red),
)

# How did we do? Did the encoding of further knowledge (conservation of drug
# between Depot and Central) make the model better?

# 2.3. Extend the analysis to a population of multiple, heterogeneous, subjects
#

data_model_heterogeneous = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower = 0.0)  # typical value of maximum inhibition
        tvIC50 ∈ RealDomain(; lower = 0.0)  # typical value of concentration for half-way inhibition
        tvKa ∈ RealDomain(; lower = 0.0)    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)       # residual error
    end
    @random η ~ MvNormal(Diagonal([0.1, 0.1, 0.1]))
    @pre begin
        Imax = tvImax * exp(η[1])
        IC50 = tvIC50 * exp(η[2])
        Ka = tvKa * exp(η[2]) 
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end


# 2.4. Analyse the effect of very sparse data on the predictions

sims_sparse = [
    simobs(
        data_model_heterogeneous,
        Subject(; events = DosageRegimen(5.0), id = i),
        true_parameters;
        obstimes = 11 .* sort!(rand(2)),
    ) for i = 1:25
]
population_sparse = Subject.(sims_sparse)
plotgrid(population_sparse)

fpm_sparse = fit(
    ude_model_knowledge,
    population_sparse,
    init_params(ude_model_knowledge),
    MAP(NaivePooled()),
)

pred = predict(fpm_sparse; obstimes = 0:0.01:10);
plotgrid(pred)

# plot them all stacked ontop of oneanother
fig = Figure();
ax = Axis(fig[1,1]; xlabel="Time", ylabel="Outcome", title="Stacked predictions")
for i in eachindex(pred)
    plotgrid!([ax], pred[i:i]; data=(; color=Cycled(i)))
end
fig

# Does it look like we've found anything reasonable?


# 2.5. Finally, what if we have multiple patients with fairly rich timecourses?

population = synthetic_data(data_model_heterogeneous, DosageRegimen(5.0), true_parameters; obstimes=0:1:10, nsubj=25, rng=StableRNG(1))
plotgrid(population)

fpm_knowledge_2 = fit(
    ude_model_knowledge,
    population,
    init_params(ude_model_knowledge),
    MAP(NaivePooled()),
)

pred = predict(fpm_knowledge_2; obstimes=0:0.1:10);
plotgrid(pred)
