using DeepPumas

@info "Compiling the machine learning exercises"
@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_01_machine_learning.jl")
  return nothing
end

@info "Compiling the neural IDR exercise"
@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_02_idr.jl")
  return nothing
end

@info "Compiling the UDE exercises"
@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_03_SciML.jl")
  return nothing
end

@info "Compiling the MeNet exercises"
@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_04_MeNets.jl")
  return nothing
end

@info "Finished compiling functions!"
