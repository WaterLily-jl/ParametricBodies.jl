module ParametricBodiesAMDGPUExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using ParametricBodies

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

AMDGPU.allowscalar(false) # disallow scalar operations on GPU

end # module
