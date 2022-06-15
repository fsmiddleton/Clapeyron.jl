"""
    PairParam{T}
Struct designed to contain pair data. used a matrix as underlying data storage.
## Creation:
```julia-repl
julia> kij = PairParam("interaction params", ["water","ammonia"], [0.1 0.0; 0.1 0.0])
PairParam{Float64}["water", "ammonia"]) with values:
2×2 Matrix{Float64}:
 0.1  0.0
 0.1  0.0
julia> kij.values
2×2 Matrix{Float64}:
 0.1  0.0
 0.1  0.0
julia> kij.diagvalues
2-element view(::Vector{Float64}, 
1:3:4) with eltype Float64:
 0.1
 0.0
```
## Example usage in models:
```julia
#lets compute ∑xᵢxⱼkᵢⱼ
function alpha(model, x)
    kij = model.params.kij.values
    ki = model.params.kij.diagvalues
    res = zero(eltype(molarfrac))
    for i in @comps 
        @show ki[i] #diagonal values
        for j in @comps 
            res += x[i] * x[j] * kij[i,j]
        end
    end
    return res
end
```
"""
struct PairParameter{T,V<:AbstractMatrix{T},D} <: ClapeyronParam
    name::String
    components::Vector{String}
    groups::Vector{String}
    grouptype::Union{Symbol,Nothing}
    values::V
    symmetric::Bool
    ismissingvalues::Matrix{Bool}
    sourcecsvs::Vector{String}
    sources::Vector{String}
end

const PairParam{T} = PairParameter{T,Matrix{T}} where T

PairParam(name, components, groups, grouptype, values, symmetric, ismissingvals, sourcecsvs, sources) = PairParameter(name, components, groups, grouptype, values, symmetric, ismissingvals, sourcecsvs, sources)

# Legacy format without groups
function PairParam(
        name::String,
        components::Vector{String},
        values::Matrix{T},
        symmetric::Bool = true,
        ismissingvalues = fill(false, length(components), length(components)),
        sourcecsvs::Vector{String} = String[], 
        sources::Vector{String} = String[]) where T
    Base.depwarn("Params should be constructed with group info.", :PairParameter; force=true)
    _values, _ismissingvalues = defaultmissing(values)
    if !all(ismissingvalues)
        _ismissingvalues = ismissingvalues
    end
    return PairParameter(
        name,
        components,
        nothing,
        _values,
        symmetric,
        _ismissingvalues,
        sourcecsvs,
        sources,
    )
end

# Indexing

Base.@propagate_inbounds Base.getindex(param::PairParameter{T,<:AbstractMatrix{T}}, i::Int) where T = param.values[i,i]
Base.@propagate_inbounds Base.getindex(param::PairParameter{T,<:AbstractMatrix{T}}, i::Int, j::Int) where T = param.values[i,j]
Base.setindex!(param::PairParameter, val, i) = setindex!(param.values, val, i, i)
function Base.setindex!(param::PairParameter, val, i, j) 
    setindex!(param.values, val, i, j)
    param.symmetric && setindex!(param.values, val, j, i)
end

# Broadcasting
Base.broadcastable(param::PairParameter) = param.values
Base.BroadcastStyle(::Type{<:PairParameter}) = Broadcast.Style{PairParameter}()

# copyto!
function Base.copyto!(param::PairParameter, x)
    Base.copyto!(param.values, x)
    return param
end

function Base.copyto!(dest::PairParameter, src::PairParameter) #used to set params
    #key check
    dest.components == src.components || throw(DimensionMismatch("components of source and destination pair parameters are not the same for $dest"))
    
    #=
    TODO: it does not check that dest.symmetric = src.symmetric, the only solution i see at the moment 
    is to make the Single, Pair and Assoc Params, mutable structs, but i don't really know the performance
    implications of that.

    supposedly, this copyto! is only used internally, and both src and dest params are already the same. but it would
    be good to enforce that.
    =#
    copyto!(dest.values, src.values)
    dest.ismissingvalues .= src.ismissingvalues
    return dest
end

Base.size(param::PairParameter) = size(param.values)
components(x::PairParameter) = x.components

# Unsafe constructor
function PairParam(
        name::String,
        components::Vector{String},
        groups::Vector{String},
        grouptype::Union{Symbol,Nothing},
        values,
        symmetric = true,
        sourcecsvs = String[],
        sources = String[],
    )
    ismissingvalues = fill(false, size(values))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        symmetric,
        ismissingvalues,
        sourcecsvs,
        sources,
    )
end

function PairParam(
        name::String,
        components::Vector{String},
        values,
        symmetric = true,
        sourcecsvs = String[],
        sources = String[],
    )
    missingvals = fill(false, size(values))
    return PairParam(
        name,
        components,
        components,
        nothing,
        values,
        symmetric,
        missingvals,
        sourcecsvs,
        sources,
    )
end

# Name changing for constructing params from inputparams
function PairParam(
        x::PairParam,
        name::String = x.name;
        isdeepcopy = true,
        sources = x.sources,
    )
    if isdeepcopy
        values = deepcopy(x.values)
        return PairParam(
            name,
            x.components,
            x.groups,
            x.grouptype,
            values,
            x.symmetric,
            deepcopy(x.ismissingvalues),
            x.sourcecsvs,
            sources,
        )
    end
    return PairParam(
        name,
        x.components,
        x.groups,
        x.grouptype,
        x.values,
        x.symmetric,
        x.ismissingvalues,
        x.sourcecsvs,
        sources,
    )
end

PairParameter(x::PairParam, name::String = x.name; isdeepcopy = true, sources = x.sources) = PairParam(x, name; isdeepcopy, sources)

# If no value is provided, just initialise empty param
function PairParam{T}(
        name::String,
        components::Vector{String};
        groups::Vector{String} = String[],
        grouptype::Union{Symbol,Nothing} = nothing,
        symmetric::Bool = true,
        sources::Vector{String} = String[],
    ) where T <: AbstractString
    values = fill("", length(components), length(components))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        symmetric,
        String[],
        sources,
    )
end

function PairParam{T}(
        name::String,
        components::Vector{String};
        groups::Vector{String} = String[],
        grouptype::Union{Symbol,Nothing} = nothing,
        symmetric = true,
        sources::Vector{String} = String[],
    ) where T <: Number
    values = zeros(T, length(components), length(components))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        symmetric,
        String[],
        sources,
    )
end


# Single to pair promotion
function PairParam(x::SingleParam, name::String = x.name, symmetric = true)
    pairvalues = singletopair(x.values, missing)
    for i in 1:length(x.values)
        if x.ismissingvalues[i]
            pairvalues[i,i] = missing
        end
    end
    _values, _ismissingvalues = defaultmissing(pairvalues)
    return PairParam(
        name,
        x.components,
        x.groups,
        x.grouptype,
        _values,
        symmetric,
        _ismissingvalues,
        x.sourcecsvs,
        x.sources,
    )
end

# Show
function Base.show(io::IO,mime::MIME"text/plain", param::PairParameter) 
    sym = param.symmetric ? "Symmetric " : ""
    _size = size(param)
    _size_str = string(_size[1]) * "×" * string(_size[2]) * " "
    print(io, sym, _size_str, "PairParam{", eltype(param.values), "}(")
    show(io, param.components)
    println(io, ") with values:")
    Base.print_matrix(IOContext(io, :compact => true), param.values)
end

function Base.show(io::IO, param::PairParameter)
    print(io, "PairParam{", eltype(param.values), "}", "(\"", param.name, "\")[")
    print(io, Base.summary(param.values))
    print(io, "]")
end

# Convert utilities
function Base.convert(::Type{PairParam{Float64}}, param::PairParam{Int})
    values = Float64.(param.values)
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.convert(::Type{PairParam{Bool}}, param::PairParam{Int})
    @assert all(z -> (isone(z) | iszero(z)), param.values)
    values = Array(Bool.(param.values))
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.convert(::Type{PairParam{Int}}, param::PairParam{Float64})
    @assert all(z -> isinteger(z), param.values)
    values = Int.(param.values)
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.convert(
        ::Type{PairParam{String}},
        param::PairParam{<:AbstractString}
    )::PairParam{String}
    values = String.(param.values)
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

# Pack vectors
function pack_vectors(param::PairParameter{<:AbstractVector})
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        pack_vectors(param.values),
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

const PackedSparsePairParam{T} = Clapeyron.PairParameter{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, SparsePackedMofV{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, 
true}, PackedVectorsOfVectors.PackedVectorOfVectors{Vector{Int64}, Vector{T}, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}}} where T

# Operations
function Base.:(+)(param::PairParameter, x::Number)
    values = param.values .+ x
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.:(*)(param::PairParameter, x::Number)
    values = param.values .* x
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.:(^)(param::PairParameter, x::Number)
    values = param.values .^ x
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.symmetric,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end
