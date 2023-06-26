using DeepPumas

@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_01_machine_learning.jl")
end

@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_02_idr.jl")
end

@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_03_SciML.jl")
end

@time redirect_stdout(devnull) do 
  include(dirname(@__FILE__()) * "/2_04_MeNets.jl")
end

@info "Finished compiling functions!"
