### A Pluto.jl notebook ###
# v0.19.12

using Pumas
using PumasPlots
using CairoMakie
using AlgebraOfGraphics

linear_model = @model begin
    @param begin
        tvkgl ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kgl = tvkgl*exp(η[1])
        TS0 = tvTS0*exp(η[2])
		
		TS = kgl*t + TS0 # Analytical solution
    end  
    @derived begin
		ts ~ Normal.(TS, σ)
    end
end;

linear_params = (;
    tvkgl=0.1,
    tvTS0=2.0,
    Ω=I(2)*1e-12,
    σ=1e-4
);

tsim = 0:0.1:100;

linear_sim = simobs(
	linear_model, 
	Subject(), 
	linear_params; 
	obstimes=tsim
);

sim_plot(
	linear_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Linear model"
	)
)

quadratic_model = @model begin
    @param begin
        tvkgl ∈ RealDomain(; lower=0)
        tvkg2 ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
         η ~ MvNormal(Ω)
    end
    @pre begin
        kgl = tvkgl*exp(η[1])
        kg2 = tvkg2*exp(η[2])
        TS0 = tvTS0*exp(η[3])

        TS = kgl*t + kg2*t^2 + TS0 # Analytical solution
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

quadratic_params = (;
    tvkgl=0.1,
    tvkg2=0.001,
    tvTS0=2.0,
    Ω=I(3)*1e-12,
    σ=1e-4
);

quadratic_sim = simobs(
	quadratic_model, 
	Subject(), 
	quadratic_params; 
	obstimes=tsim
);

sim_plot(
	quadratic_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Quadratic model"
	)
)

exponential_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        TS0 = tvTS0*exp(η[2])

        TS = TS0*exp(kge*t) # Analytical solution
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

exponential_params = (;
    tvkge=0.029,
    tvTS0=2.0,
    Ω=I(2)*1e-12,
    σ=1e-4
);

exponential_sim = simobs(
	exponential_model, 
	Subject(), 
	exponential_params; 
	obstimes=tsim
);

exponential_fig = Figure(; resolution=(1200, 600));
sim_plot(
	exponential_fig[1, 1],
	exponential_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Exponential model - linear scale"
	)
);
sim_plot(
	exponential_fig[1, 2],
	exponential_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Exponential model - logarithmic scale",
		yscale=log, # Logarithmic scale
		yticks=2500:2500:12500
	)
);
exponential_fig

power_law_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvγ ∈ RealDomain(; lower=0, upper=1)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        γ = tvγ*exp(η[2])
        TS0 = tvTS0*exp(η[3])

        γ2 = 1 - γ
        TS = (kge*γ2*t + TS0^γ2)^(1/γ2) # Analytical solution
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

power_params = (;
    tvγ=0.8,
    tvkge=0.029,
    tvTS0=2.0,
    Ω=I(3)*1e-12,
    σ=1e-4
);

power_sim = simobs(
	power_law_model, 
	Subject(), 
	power_params; 
	obstimes=tsim
);

sim_plot(
	power_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Power law"
	)
)

explin_model = @model begin
    @param begin
        tvkgl ∈ RealDomain(; lower=0)
        tvkge ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kgl = tvkgl*exp(η[1])
        kge = tvkge*exp(η[2])
        TS0 = tvTS0*exp(η[3])

        τ = (1/kge)*log(kgl/(kge*TS0))
        TS = t <= τ ? TS0*exp(kge*t) : kgl*(t - τ) + TS0*exp(kge*τ)
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

explin_params = (;
    tvkgl=0.2,
    tvkge=0.029,
    tvTS0=2.0,
    Ω=I(3)*1e-12,
    σ=1e-4
);

explin_sim = simobs(
	explin_model,
	Subject(),
	explin_params;
	obstimes=tsim
);

sim_plot(
	explin_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Exponential-linear",
	)
)

simeoni_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvkgl ∈ RealDomain(; lower=0)
        tvψ ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        kgl = tvkgl*exp(η[2])
        ψ = tvψ*exp(η[3])
        TS0 = tvTS0*exp(η[4])
    end
    @init TS = TS0
    @dynamics begin
        TS' = kge*TS / (1 + ((kge/kgl)*TS)^ψ)^(1/ψ)
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

simeoni_params = (;
    tvψ=1300,
    tvkge=0.02,
    tvkgl=0.2,
    tvTS0=2.0,
    Ω=I(4)*1e-12,
    σ=1e-4
);

simeoni_sim = simobs(
	simeoni_model, 
	Subject(), 
	simeoni_params; 
	obstimes=tsim
);

sim_plot(
	simeoni_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Simeoni"
	)
)

koch_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvkgl ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        kgl = tvkgl*exp(η[2])
        TS0 = tvTS0*exp(η[3])
    end
    @init TS = TS0
    @dynamics begin
        TS' = (2kge*kgl*TS) / (kgl + 2kge*TS)
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

koch_params = (;
    tvkge=0.011,
    tvkgl=116,
    tvTS0=2.0,
    Ω=I(3)*1e-12,
    σ=1e-4
);

koch_sim = simobs(
	koch_model, 
	Subject(), 
	koch_params; 
	obstimes=tsim
);

sim_plot(
	koch_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Koch"
	)
)

logistic_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvTSmax ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        TS0 = tvTS0*exp(η[2])
        TSmax = tvTSmax*exp(η[3])

        TS = (TSmax*TS0) / (TS0 + (TSmax - TS0)*exp(-kge*t))
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

logistic_params = (;
    tvkge=0.07,
    tvTS0=2.0,
    tvTSmax=30,
    Ω=I(3)*1e-12,
    σ=1e-4
);

logistic_sim = simobs(
	logistic_model, 
	Subject(), 
	logistic_params; 
	obstimes=tsim
);

sim_plot(
	logistic_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Logistic"
	)
)

gen_logistic_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvTSmax ∈ RealDomain(; lower=0)
        tvγ ∈ RealDomain(; lower=0, upper=1)
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        TS0 = tvTS0*exp(η[2])
        TSmax = tvTSmax*exp(η[3])
        γ = tvγ*exp(η[4])

        TS = (TSmax*TS0) / (TS0^γ + (TSmax^γ - TS0^γ)*exp(-kge*γ*t))^(1/γ)
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

gen_logistic_params = (;
    tvkge=0.06,
    tvTS0=2.0,
    tvTSmax=30,
    tvγ=0.9,
    Ω=I(4)*1e-12,
    σ=1e-4
);

gen_logistic_sim = simobs(
	gen_logistic_model, 
	Subject(), 
	gen_logistic_params; 
	obstimes=tsim
);

sim_plot(
	gen_logistic_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Generalized logistic"
	)
)

simeoni_log_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvkgl ∈ RealDomain(; lower=0)
        tvψ ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvTSmax ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(5)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        kge = tvkge*exp(η[1])
        kgl = tvkgl*exp(η[2])
        ψ = tvψ*exp(η[3])
        TS0 = tvTS0*exp(η[4])
        TSmax = tvTSmax*exp(η[5])
    end
    @init TS = TS0
    @dynamics begin
        TS' = kge*TS*(1 - (TS/TSmax)) / ((1 + (TS*kge/kgl)^ψ)^(1/ψ))
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

simeoni_log_params = (;
    tvψ=520,
    tvkge=0.04,
    tvkgl=104,
    tvTS0=2.0,
    tvTSmax=30.0,
    Ω=I(5)*1e-12,
    σ=1e-4
);

simeoni_log_sim = simobs(
	simeoni_log_model, 
	Subject(), 
	simeoni_log_params; 
	obstimes=tsim
);

sim_plot(
	simeoni_log_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Simeoni-logistic"
	)
)

gompertz_model = @model begin
    @param begin
        tvβ ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvTSmax ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        β = tvβ*exp(η[1])
        TS0 = tvTS0*exp(η[2])
        TSmax = tvTSmax*exp(η[3])

        # Analytical solution
		TS = TSmax*(TS0/TSmax)^exp(-β*t)
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

gompertz_params = (;
    tvβ=0.08,
    tvTS0=2.0,
    tvTSmax=30.0,
    Ω=I(3)*1e-12,
    σ=1e-4
);

gompertz_sim = simobs(
	gompertz_model, 
	Subject(), 
	gompertz_params; 
	obstimes=tsim
);

sim_plot(
	gompertz_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Gompertz"
	)
)

exp_gompertz_model = @model begin
    @param begin
        tvkge ∈ RealDomain(; lower=0)
        tvβ ∈ RealDomain(; lower=0)
        tvTS0 ∈ RealDomain(; lower=0)
        tvTSmax ∈ RealDomain(; lower=0)
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(; lower=0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @init TS = TS0
    @pre begin
        kge = tvkge*exp(η[1])
        β = tvβ*exp(η[2])
        TS0 = tvTS0*exp(η[3])
        TSmax = tvTSmax*exp(η[4])
    end
    @dynamics begin
        TS' = min(kge*TS, TS*β*log(TSmax/TS))
    end
    @derived begin
        ts ~ Normal.(TS, σ)
    end
end;

exp_gompertz_params = (;
    tvβ=0.11,
    tvkge=0.054,
    tvTS0=2.0,
    tvTSmax=30.0,
    Ω=I(4)*1e-12,
    σ=1e-4
);

exp_gompertz_sim = simobs(
	exp_gompertz_model, 
	Subject(), 
	exp_gompertz_params; 
	obstimes=tsim
);

sim_plot(
	exp_gompertz_sim;
	axis=(;
		ylabel="Sum of Longest Diameters (cm)",
		xlabel="Time (weeks)",
		title="Exponential Gompertz"
	)
)
