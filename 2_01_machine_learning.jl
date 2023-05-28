using DeepPumas
using CairoMakie
using Distributions
using Random

macro preprocess(_x, _y)
    :(DeepPumas.FitTarget{$(QuoteNode(_x)),$(QuoteNode(_y))}(
        $(_x),
        $(_y),
        identity,
        identity,
    ))
end
(f::DeepPumas.HoldoutValidationResult)(args...) = f.ml.ml(args...)
(f::DeepPumas.HyperoptResult)(args...) = f.ml.ml(args...)
(f::DeepPumas.SimpleChainDomain)(args...) = f.init(args...)
(f::DeepPumas.FluxDomain)(args...) = f.init(args...)

# 
# TABLE OF CONTENTS
# 
#
# 0. BASIC SAMPLE PARAMETERS 
#
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
#
# 1.1. Sample subjects with an obvious `true_function`
# 1.2. Model `true_function` with a linear regression
#
# 2. CAPTURING COMPLEX RELATIONSHIPS
#
# 2.1. Sample subjects with a more complex `true_function`
# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`
# 2.3. Use a neural network (NN) to model `true_function`
#
# 3. BASIC UNDERFITTING AND OVERFITTING
#
# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train again the NN for few and for many iterations.)
# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
# 3.3. The impact of the NN size
# 
# 4. INSPECTION OF THE VALIDATION LOSS AND REGULARIZATION
#
# 4.1. Validation loss as a proxy for generalization performance
# 4.2. Regularization to prevent overfitting
# 4.3. Hyperparameter tuning
# 

#
# 0. BASIC SAMPLE PARAMETERS
#

num_samples = 100
uniform = Uniform(-1, 1)
normal = Normal(0, 1)
σ = 0.25

#
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
#
# 1.1. Sample from an obvious `true_function`
# 1.2. Model `true_function` with a DeepPumas linear regression
#

# 1.1. Sample from an obvious `true_function`

true_function = x -> x
x = rand(uniform, 1, num_samples)   # samples stored columnwise
ϵ = rand(normal, 1, num_samples)    # samples stored columnwise
y = true_function.(x) + σ * ϵ

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 1.2. Model `true_function` with a linear regression

target = @preprocess x y                        # DeepPumas `target`
linreg = MLP(1, (1, identity); bias = true)     # DeepPumas multilayer perceptron 
# TODO Make this manually as a SimpleChain?

fitted_linreg = fit(linreg, target; optim_alg = DeepPumas.BFGS());
fitted_linreg  # TODO throws
fitted_linreg.ml.ml.param   # `true_function` is y = x (that is, a = 1 b = 0)
# TODO Better way to show params?

ŷ = fitted_linreg(x)

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ); label = "prediction")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 2. CAPTURING COMPLEX RELATIONSHIPS
#
# 2.1. Sample from a more complex `true_function`
# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`
# 2.3. Use a neural network (NN) to model `true_function`
#

# 2.1. Sample from a more complex `true_function`

true_function = x -> x^2
x = rand(uniform, 1, num_samples)
ϵ = rand(normal, 1, num_samples)
y = true_function.(x) + σ * ϵ

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 2.2. Exercise: Reason about using a linear regression to model `true_function`

target = @preprocess x y
fitted_linreg =
    fit(linreg, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 50));
fitted_linreg  # TODO throws
fitted_linreg.ml.ml.param

ŷ_ex22_50iter = fitted_linreg(x)

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ_ex22_50iter); label = "prediction")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 2.3. Use a neural network (NN) to model `true_function`

nn = MLP(1, (8, tanh), (1, identity); bias = true)
fitted_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 50));
fitted_nn  # TODO throws  
fitted_nn.ml.ml.param  # try to make sense of the parameters in the NN

ŷ = fitted_nn(x)

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ), label = "prediction")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 3. BASIC UNDERFITTING AND OVERFITTING
#
# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train again the NN for few and for many iterations.)
# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
# 3.3. The impact of the NN size
# 

# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train again the NN for few and for many iterations.)

underfit_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 5));
ŷ_underfit = underfit_nn(x)

overfit_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 1_000));
ŷ_overfit = overfit_nn(x)  # clarification on the term "overfitting"

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ_underfit), label = "prediction (5 iterations)")
    scatter!(vec(x), vec(ŷ), label = "prediction (50 iterations)")
    scatter!(vec(x), vec(ŷ_overfit), label = "prediction (1000 iterations)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
#      Investigate the effect of `max_iterations`.

begin
    max_iterations = 2
    fitted_linreg = fit(
        linreg,
        target;
        optim_alg = DeepPumas.BFGS(),
        optim_options = (; iterations = max_iterations),
    )
    ŷ_linreg = fitted_linreg(x)

    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ_linreg), label = "$max_iterations iterations")
    scatter!(vec(x), vec(ŷ_ex22_50iter), label = "50 iterations")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.3. The impact of the NN size

nn = MLP(1, (32, tanh), (32, tanh), (1, identity); bias = true)
fitted_nn = fit(
    nn, 
    target; 
    optim_alg = DeepPumas.BFGS(), 
    optim_options = (; iterations = 1_000)
);
fitted_nn # TODO throws

ŷ = fitted_nn(x)

begin
    f = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data")
    scatter!(vec(x), vec(ŷ), label = "prediction MLP(1, 32, 32, 1)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 4. INSPECTION OF THE VALIDATION LOSS, REGULARIZATION AND HYPERPARAMETER TUNING
#
# 4.1. Validation loss as a proxy for generalization performance
# 4.2. Regularization to prevent overfitting
# 4.3. Hyperparameter tuning
# 

# 4.1. Validation loss as a proxy for generalization performance

x_train, y_train = x, y
target_train = target

ϵ = rand(normal, 1, num_samples)
x_valid = rand(uniform, 1, num_samples)
y_valid = true_function.(x_valid) + σ * ϵ
target_valid = @preprocess x_valid y_valid

begin
    f = scatter(
        vec(x_train),
        vec(y_train);
        axis = (xlabel = "x", ylabel = "y"),
        label = "training data",
    )
    scatter!(vec(x_valid), vec(y_valid); label = "validation data")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

begin
    loss_train_l, loss_valid_l = [], []

    fitted_nn = fit(
        nn,
        target_train;
        optim_alg = DeepPumas.BFGS(),
        optim_options = (; iterations = 10),
    )
    push!(loss_train_l, sum((fitted_nn(x_train) .- y_train) .^ 2))
    push!(loss_valid_l, sum((fitted_nn(x_valid) .- y_valid) .^ 2))

    iteration_blocks = 100
    for _ = 2:iteration_blocks
        fitted_nn = fit(
            nn,
            target_train,
            fitted_nn.ml.ml.param;
            optim_alg = DeepPumas.BFGS(),
            optim_options = (; iterations = 10),
        )
        push!(loss_train_l, sum((fitted_nn(x_train) .- y_train) .^ 2))
        push!(loss_valid_l, sum((fitted_nn(x_valid) .- y_valid) .^ 2))
    end
end

begin
    f, ax = scatterlines(
        1:iteration_blocks,
        Float32.(loss_train_l);
        label = "training",
        axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Mean squared loss"),
    )
    scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
    axislegend()
    f
end

# 4.2. Regularization to prevent overfitting

reg_nn = MLP(1, (32, tanh), (32, tanh), (1, identity); bias = true, reg = L2(0.1))

begin
    reg_loss_train_l, reg_loss_valid_l = [], []

    fitted_reg_nn = fit(
        reg_nn,
        target_train;
        optim_alg = DeepPumas.BFGS(),
        optim_options = (; iterations = 10),
    )
    push!(reg_loss_train_l, sum((fitted_reg_nn(x_train) .- y_train) .^ 2))
    push!(reg_loss_valid_l, sum((fitted_reg_nn(x_valid) .- y_valid) .^ 2))

    iteration_blocks = 100
    for _ = 2:iteration_blocks
        fitted_reg_nn = fit(
            reg_nn,
            target_train,
            fitted_reg_nn.ml.ml.param;
            optim_alg = DeepPumas.BFGS(),
            optim_options = (; iterations = 10),
        )
        push!(reg_loss_train_l, sum((fitted_reg_nn(x_train) .- y_train) .^ 2))
        push!(reg_loss_valid_l, sum((fitted_reg_nn(x_valid) .- y_valid) .^ 2))
    end
end

begin
    f, ax = scatterlines(
        1:iteration_blocks,
        Float32.(loss_train_l);
        label = "training",
        axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Mean squared loss"),
    )
    scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
    scatterlines!(1:iteration_blocks, Float32.(reg_loss_train_l); label = "training (L2)")
    scatterlines!(1:iteration_blocks, Float32.(reg_loss_valid_l); label = "validation (L2)")
    axislegend()
    f
end

# 4.3. Hyperparameter tuning

#TODO Add hyperopt
## Fit with many different hyperparameters to optimize for generalization performance
nn_ho = hyperopt(reg_nn, target_train)
ŷ_ho = nn_ho(x_valid)

begin
    f = scatter(vec(x_valid), vec(y_valid); label = "validation data")
    scatter!(vec(x_valid), vec(ŷ_ho), label = "prediction (hyperparam opt.)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end
