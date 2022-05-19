"""
    SingleParam{T}

Struct designed to contain single parameters. Basically a vector with some extra info.

## Creation:
```julia-repl
julia> mw = SingleParam("molecular weight",["water","ammonia"],[18.01,17.03])
SingleParam{Float64}("molecular weight") with 2 components:
 "water" => 18.01
 "ammonia" => 17.03

julia> mw.values
2-element Vector{Float64}:
 18.01
 17.03

julia> mw.components
2-element Vector{String}:
 "water"
 "ammonia"

julia> mw2 = SingleParam(mw,"new name")
SingleParam{Float64}("new name") with 2 components:
 "water" => 18.01
 "ammonia" => 17.03

julia> has_oxygen = [true,false]; has_o = SingleParam(mw2, has_oxygen)
SingleParam{Bool}("new name") with 2 components:
 "water" => true
 "ammonia" => false

```

## Example usage in models:

```
function molecular_weight(model, molar_frac)
    mw = model.params.mw.values
    res = zero(eltype(molarfrac))
    for i in @comps  # iterating through all components
        res += molar_frac[i] * mw[i]
    end
    return res
end
```
"""
struct SingleParameter{T,V<:AbstractVector{T}} <: ClapeyronParam
    name::String
    components::Vector{String}
    groups::Vector{String}
    grouptype::Union{Symbol,Nothing}
    values::V
    ismissingvalues::Vector{Bool}
    sourcecsvs::Vector{String}
    sources::Vector{String}
end

const SingleParam{T} = SingleParameter{T,Vector{T}} where T

SingleParam(name, components, groups, grouptype, values, missingvals, src, sourcecsv) = SingleParameter(name, components, groups, grouptype, values, missingvals, src, sourcecsv)

# Group-less constructor for legacy purposes
function SingleParam(
        name::String,
        components::Vector{String},
        values::Vector{T},
        ismissingvalues::Vector{Bool},
        sourcecsvs,
        sources,
    ) where T
    Base.depwarn("Params should be constructed with group info.", :SingleParameter; force=true)
    return SingleParam{T}(
        name,
        components,
        components,
        nothing,
        values,
        ismissingvalues,
        sourcecsvs,
        sources,
    )
end

# Constructor to fill in ismissingvalues
function SingleParam(
        name::String,
        components::Vector{String},
        groups::Vector{String},
        grouptype::Union{Symbol,Nothing},
        values::Vector{T},
        sourcecsvs = String[],
        sources = String[],
    ) where T
    if any(ismissing, values)
        _values, _ismissingvalues = defaultmissing(values)
        TT = eltype(_values)
    else
        _values = values
        _ismissingvalues = fill(false, length(values))
        TT = T
    end
    return SingleParam{TT}(
        name,
        components,
        groups,
        grouptype,
        _values,
        _ismissingvalues,
        sourcecsvs,
        sources,
    )
end

# Legacy format without groups
function SingleParam(
        name::String,
        components::Vector{String},
        values::Vector{T},
        sourcecsvs = String[],
        sources = String[],
    ) where T
    Base.depwarn("Params should be constructed with group info.", :SingleParameter; force=true)
    if any(ismissing, values)
        _values,_ismissingvalues = defaultmissing(values)
        TT = eltype(_values)
    else
        _values = values
        _ismissingvalues = fill(false, length(values))
        TT = T
    end
    return SingleParam{TT}(
        name,
        components,
        components,
        nothing,
        _values,
        _ismissingvalues,
        sourcecsvs,
        sources,
    )
end

# Name changing for constructing params from inputparams
function SingleParam(
        x::SingleParam,
        name::String = x.name;
        isdeepcopy::Bool = true,
        sources::Vector{String} = x.sources,
    )
    if isdeepcopy
        return SingleParam(
            name,
            x.components,
            x.groups,
            x.grouptype,
            deepcopy(x.values),
            deepcopy(x.ismissingvalues),
            x.sourcecsvs,
            sources,
        )
    end
    return SingleParam(
        name,
        x.components,
        x.groups,
        x.grouptype,
        x.values,
        x.ismissingvalues,
        x.sourcecsvs,
        sources,
    )
end

SingleParameter(x::SingleParam, name::String = x.name; isdeepcopy::Bool = true, sources::Vector{String} = x.sources) = SingleParam(x, name; isdeepcopy, sources)

# If no value is provided, just initialise empty param
function SingleParam{T}(
        name::String,
        components::Vector{String};
        groups::Vector{String} = String[],
        grouptype::Union{Symbol,Nothing} = nothing,
        sources = String[],
    ) where T <: AbstractString
    values = fill("", length(components))
    return SingleParam(
        name,
        components,
        groups,
        grouptype,
        values,
        String[],
        sources,
    )
end

function SingleParam{T}(
        name::String,
        components::Vector{String};
        groups::Vector{String} = String[],
        grouptype::Union{Symbol,Nothing} = nothing,
        sources = String[],
    ) where T <: Number
    values = zeros(T, length(components))
    return SingleParam(
        name,
        components,
        groups,
        grouptype,
        values,
        String[],
        sources,
    )
end

# # Create copy with values replaced
function SingleParam(x::SingleParameter, v::Vector)
    _values, _ismissingvalues = defaultmissing(v)
    return SingleParam(
        x.name,
        x.components,
        x.groups,
        x.grouptype,
        _values,
        _ismissingvalues,
        x.sourcecsvs,
        x.sources,
    )
end

# Show
function Base.show(io::IO, param::SingleParameter)
    print(io, typeof(param), "(\"", param.name, "\")[")
    for component in param.components
        component != first(param.components) && print(io, ",")
        print(io, "\"", component, "\"")
    end
    print(io, "]")
end

function Base.show(io::IO, ::MIME"text/plain", param::SingleParameter)
    len = length(param.values)
    print(io, "SingleParam{",eltype(param.values), "}(\"", param.name)
    println(io, "\") with ", len, " component", ifelse(len==1, ":", "s:"))
    i = 0
    for (name, val, miss) in zip(param.components, param.values, param.ismissingvalues)
        i += 1
        if i > 1
            println(io)
        end
        if miss == false
            if typeof(val) <: AbstractString
                print(io, " \"", name, "\" => \"", val, "\"")
            else
                print(io, " \"", name, "\" => ", val)
            end
        else
            print(io, " \"", name, " => -")
        end
    end
end

# Convert utilities
function Base.convert(::Type{SingleParam{Float64}}, param::SingleParam{Int})
    values = Float64.(param.values)
    return SingleParam(
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

function Base.convert(::Type{SingleParam{Bool}}, param::SingleParam{Int})
    @assert all(z -> (isone(z) | iszero(z)), param.values)
    values = Array(Bool.(param.values))
    return SingleParam(
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

function Base.convert(::Type{SingleParam{Int}}, param::SingleParam{Float64})
    @assert all(z -> isinteger(z), param.values)
    values = Int.(param.values)
    return SingleParam(
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

function Base.convert(
        ::Type{SingleParam{String}},
        param::SingleParam{<:AbstractString}
    )::SingleParam{String}
    values = String.(param.values)
    return SingleParam(
        param.name,
        param.components,
        param.groups,
        param.grouptype,
        values,
        param.missingvals,
        param.src,
        param.sourcecsv
    )
end

# Broadcasting utilities
Base.broadcastable(param::SingleParameter) = param.values

# Pack vectors
const PackedVectorSingleParam{T} = Clapeyron.SingleParameter{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, PackedVectorsOfVectors.PackedVectorOfVectors{Vector{Int64}, Vector{T}, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}}

function pack_vectors(param::SingleParameter{<:AbstractVector})
    values = pack_vectors(vals)
    return SingleParam(
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

function pack_vectors(params::Vararg{SingleParameter{T},N}) where {T<:Number,N}
    param = first(params)
    len = length(params)
    vals = [zeros(len) for _ in params]
    for i in 1:length(vals)
        vali = vals[i]
        for (k,par) in pairs(params)
            vali[k] = par.values[i]
        end
    end
    vals = PackedVectorsOfVectors.pack(vals)
    return SingleParam(
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

# Operations
function Base.:(+)(param::SingleParameter, x::Number)
    values = param.values .+ x
    return SingleParam(
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

function Base.:(*)(param::SingleParameter, x::Number)
    values = param.values .* x
    return SingleParam(
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

function Base.:(^)(param::SingleParameter, x::Number)
    values = param.values .^ x
    return SingleParam(
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
