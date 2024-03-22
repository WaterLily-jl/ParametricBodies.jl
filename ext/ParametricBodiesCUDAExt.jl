module ParametricBodiesCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using ParametricBodies

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

CUDA.allowscalar(false) # disallow scalar operations on GPU

end # module
