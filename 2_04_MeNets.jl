using Random
using Distributions
using DeepPumas
using CairoMakie

## Generate data
## The "observations", Y, are functions of time but there is within-subject correlation

datamodel_me = @model begin
  @param σ ∈ RealDomain(; lower=0., init=0.05)
  @random begin
    c1 ~ Uniform(0.5, 1.5)
    c2 ~ Uniform(-2, 0)
  end
  @derived begin
    X := @. c1 * t / (t + exp10(c2))
    Y ~ @. Normal(X, σ)
  end
end

trainpop_me = synthetic_data(datamodel_me, (; σ=0.05); obstimes=0:0.05:1, nsubj=100)
testpop_me = synthetic_data(datamodel_me, (; σ=0.05); obstimes=0:0.05:1, nsubj=12)

plotgrid(trainpop_me[1:12])


## "Traditional" neural network
## Assume Y function of time but otherwise independently distributed.
model_t = @model begin
  @param begin
    NN ∈ MLPDomain(1, 4, 4, (1, identity); reg=L2())
    σ ∈ RealDomain(; lower=0.)
  end
  @pre X = NN(t)[1]
  @derived Y ~ Normal.(X, σ)
end

fpm_t = fit(model_t, trainpop_me, init_params(model_t), MAP(NaivePooled()))

pred = predict(fpm_t; obstimes=0:0.01:1)[1]

df = DataFrame(trainpop_me);
fig, ax, plt = scatter(df.time, df.Y; label="Data", axis=(;xlabel="Time", ylabel="Y"));
lines!(ax, collect(pred.time), pred.pred.Y; color=Cycled(2), label="NN prediction")
axislegend(ax; position=:lt)
fig
## Looks like we've successfully captured how the outcome Y depends on time, right?

## But, wait. We've ignored that the observations are correlated not only with
## time but with the speficic patient we're sampling from.
plotgrid(predict(fpm_t; obstimes=0:0.01:1)[1:12]; ylabel = "Y (Training data)")


## Mixed-effects neural network
model_me = @model begin
  @param begin
    NN ∈ MLPDomain(3, 6, 6, (1, identity); reg=L2(; input=false))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(0.1 .* I(2))
  @pre X = NN(t, η)[1]
  @derived Y ~ @. Normal(X, σ)
end

fpm_me = fit(
  model_me,
  trainpop_me,
  init_params(model_me),
  MAP(FOCE()); 
  optim_options=(; iterations=200),
)

# Plot training performance
pred_train = predict(fpm_me; obstimes=0:0.01:1)[1:12] 
plotgrid(pred_train)

# Plot test performance
pred_test = predict(model_me, testpop_me[1:12], coef(fpm_me); obstimes=0:0.01:1)
plotgrid(pred_test ; ylabel="Y (Test data)")

#=
When we generated the data, we sampled two parameter values for each Subject. 
In this Mixed-effect neural network, we're using two random effects, so there is hope that
the model can perfectly capture this two-dimensional between-subject variability.

But, what if our model had fewer random effects than there are dimensions of subject outcome variability?

Exercise: 
Modify the model_me to only use a single random effect (change the I(2) in the
random effects and the number of inputs in the MLP) and see how the resulting
fit does.
=#


############################################################################################
############################################################################################


#=
The quality of the fits here depend a on a few different things. Among these are:

- The number of training subjects
- The number of observations per subject
- The noisiness of the data

You may not need many patients to train on if your data is good - use and
modify the code just below to try it!

But if the data quality is a bit off, then data quantity might compensate 
  - increase σ and re-run
  - increase obstimes density or nsubj and rerun again. 
=#

traindata_new = synthetic_data(
  datamodel_me,
  (; σ=0.1);         # Tune the additive noise
  obstimes=0:0.05:1, # Modify the observation times
  nsubj=10,          # Change the number of patients
)

plotgrid(traindata_new)


@time fpm_me_2 = fit(
  model_me,
  traindata_new,
  sample_params(model_me),
  MAP(FOCE()); 
  optim_options=(; iterations=300, time_limit=3*60),
)

# Plot training performance
pred_train = predict(fpm_me_2; obstimes=0:0.01:1)[1:min(12, end)]
plotgrid(pred_train; ylabel = "Y (training data)")

# Plot test performance
pred_test = predict(model_me, testpop_me, coef(fpm_me_2); obstimes=0:0.005:1)
plotgrid(pred_test; ylabel="Y (Test data)")


#=

Another important factor is the 'dimensionality' of outcome heteroneniety in
the data versus in the model. 

Here, the synthetic data has inter-patient variability in c1 and c2. These two
parameters are linearly independent of oneanother and they affect patient
trajectories in different ways. A change in c1 cannot be compensated by a
change in c2. There are, thus, two distinct dimensions of inter-patient
variability in our synthetic data.

`model_me` is given a two-dimensional vector of independent random effects. The
model thus also has the ability to account for outcome variability along two
dimensions. The neural network will have to figure out just what these
dimension should be and how to optimally utilize the random effects it is
given, but all of that is taken care of during the fit.

So, since the model and the data has the same dimensionality of inter-patient
variability, the model should be able to make perferct ipreds if it was trained
on enough, high-quality, data.

But, what if we have a model with fewer dimensions of inter-patient variability
than our data has? The easiest way for us to play with this here is to reduce
the number of random effects we feed to the neural network in our model_me.

The model is then too 'simple' to be able to prefectly fit the data, but in
what way will it fail, and how much? Train such a model on nice and clean Data
to be able to see in what way the fit fails

=#

model_me2 = @model begin
  @param begin
    NN ∈ MLPDomain(2, 6, 6, (1, identity); reg=L2(; input=false)) # We now only have 2 inputs as opposed to 3 in model_me
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(0.1 .* I(2))
  @pre X = NN(t, η)[1]
  @derived Y ~ Normal.(X, σ)
end

great_data = synthetic_data(datamodel_me, (; σ=0.01); obstimes=0:0.05:1, nsubj=50)

plotgrid(great_data)

fpm_me2 = fit(
  model_me2,
  great_data,
  sample_params(model_me2),
  MAP(FOCE()); 
  optim_options=(; iterations=300, time_limit=3*60),
)

pred_train = predict(fpm_me2; obstimes=0:0.01:1)[1:min(12, end)]
plotgrid(pred_train; ylabel = "Y (training data)")


#=
The model *should* not be able to make a perfect fit here - but how did it do?
What aspect of the data did it seem not to capture?
=#