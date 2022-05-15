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
    diagvalues::D
    ismissingvalues::Matrix{Bool}
    sourcecsvs::Vector{String}
    sources::Vector{String}
end

const PairParam{T} = PairParameter{T, Matrix{T}, SubArray{T, 1, Vector{T}, Tuple{StepRange{Int64, Int64}}, true}} where T

PairParam(name, components, groups, grouptype, values, diagvals, missingvals, src, sourcecsv) = PairParameter(name, components, groups, grouptype, values, diagvals, missingvals, src, sourcecsv)

# Legacy format without groups
function PairParam(
        name::String,
        components::Vector{String},
        values::Matrix{T},
        ismissingvalues = fill(false, length(components), length(components)),
        sourcecsvs::Vector{String} = String[], 
        sources::Vector{String} = String[]) where T
    Base.depwarn("Params should be constructed with group info.", :PairParameter; force=true)
    _values, _ismissingvalues = defaultmissing(values)
    diagvalues = view(_values, diagind(_values))
    if !all(ismissingvalues)
        _ismissingvalues = ismissingvalues
    end
    return PairParam(
        name,
        components,
        String[],
        nothing,
        _values,
        diagvalues,
        _ismissingvalues,
        sourcecsvs,
        sources,
    )
end

# Unsafe constructor
function PairParam(
        name::String,
        components::Vector{String},
        groups::Vector{String},
        grouptype::Union{Symbol,Nothing},
        values,
        sourcecsvs = String[],
        sources = String[],
    )
    missingvals = fill(false, size(values))
    diagvals = view(values, diagind(values))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        diagvals,
        missingvals,
        sourcecsvs,
        sources,
    )
end

function PairParam(
        name::String,
        components::Vector{String},
        values,
        sourcecsvs = String[],
        sources = String[],
    )
    missingvals = fill(false, size(values))
    diagvals = view(values, diagind(values))
    return PairParam(
        name,
        components,
        String[],
        nothing,
        values,
        diagvals,
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
        diagvalues = view(values, diagind(values))
        return PairParam(
            name,
            x.components,
            x.groups,
            x.grouptype,
            values,
            diagvalues,
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
        x.diagvalues,
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
        sources::Vector{String} = String[],
    ) where T <: AbstractString
    values = fill("", length(components), length(components))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        String[],
        sources,
    )
end

function PairParam{T}(
        name::String,
        components::Vector{String};
        groups::Vector{String} = String[],
        grouptype::Union{Symbol,Nothing} = nothing,
        sources::Vector{String} = String[],
    ) where T <: Number
    values = zeros(T, length(components), length(components))
    return PairParam(
        name,
        components,
        groups,
        grouptype,
        values,
        String[],
        sources,
    )
end


# Single to pair promotion
function PairParam(x::SingleParam, name::String=x.name)
    pairvalues = singletopair(x.values, missing)
    for i in 1:length(x.values)
        if x.ismissingvalues[i]
            pairvalues[i,i] = missing
        end
    end
    _values,_ismissingvalues = defaultmissing(pairvalues)
    diagvalues = view(_values, diagind(_values))
    return PairParam(
        name,
        x.components,
        x.groups,
        x.grouptype,
        _values,
        diagvalues,
        _ismissingvalues,
        x.sourcecsvs,
        x.sources,
    )
end

# Show
function Base.show(io::IO, mime::MIME"text/plain", param::PairParameter) 
    print(io, "PairParam{" ,eltype(param.values) ,"}")
    show(io, param.components)
    println(io, ") with values:")
    show(io, mime, param.values)
end

function Base.show(io::IO, param::PairParameter)
    print(io, "PairParam{", eltype(param.values), "}", "(\"", param.name, "\")[")
    print(io, Base.summary(param.values))
    print(io, "]")
end

# Convert utilities
function Base.convert(::Type{PairParam{Float64}}, param::PairParam{Int})
    values = Float64.(param.values)
    diagvalues = view(values, diagind(values))
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        diagvalues,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.convert(::Type{PairParam{Bool}}, param::PairParam{Int})
    @assert all(z -> (isone(z) | iszero(z)), param.values)
    values = Array(Bool.(param.values))
    diagvalues = view(values, diagind(values))
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        diagvalues,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

function Base.convert(::Type{PairParam{Int}}, param::PairParam{Float64})
    @assert all(z -> isinteger(z), param.values)
    values = Int.(param.values)
    diagvalues = view(values, diagind(values))
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        diagvalues,
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
    diagvalues = view(values, diagind(values))
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        diagvalues,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

# Broadcasting utilities
Base.broadcastable(param::PairParameter) = param.values

# Pack vectors
function pack_vectors(param::PairParameter{<:AbstractVector})
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        pack_vectors(param.values),
        nothing,
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end

const PackedSparsePairParam{T} = Clapeyron.PairParameter{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, SparsePackedMofV{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, 
true}, PackedVectorsOfVectors.PackedVectorOfVectors{Vector{Int64}, Vector{T}, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}}, Nothing} where T

# Operations
function Base.:(+)(param::PairParameter, x::Number)
    values = param.values .+ x
    return PairParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
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
        param.ismissingvalues,
        param.sourcecsvs,
        param.sources,
    )
end
