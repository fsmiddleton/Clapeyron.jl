@enum CSVType singledata pairdata assocdata groupdata

const NumberOrString = Union{Union{T1,Missing},Union{T2,Missing}} where {T1 <: AbstractString, T2 <: Number}

"""
    getfileextension(filepath)

A quick helper to get the file extension of any given path (without the dot).

# Examples
```julia-repl
julia> getfileextension("~/Desktop/text.txt")
"txt"
```
""" 
function getfileextension(filepath::AbstractString)
    path, ext = splitext(filepath)
    return String(chop(ext, head=1, tail=0))
end

"""
    getpaths(location; relativetodatabase=false)

Returns database paths that is optionally relative to Clapeyron.jl directory.
If path is a file, then return an Array containing a single path to that file.
If path is a directory, then return an Array containing paths to all csv files in that directory.

# Examples
```julia-repl
julia> getpaths("SAFT/PCSAFT"; relativetodatabase=true)
3-element Array{String,1}:
 "/home/user/.julia/packages/Clapeyron.jl/xxxxx/database/SAFT/PCSAFT/data_PCSAFT_assoc.csv"
 "/home/user/.julia/packages/Clapeyron.jl/xxxxx/database/SAFT/PCSAFT/data_PCSAFT_like.csv"
 "/home/user/.julia/packages/Clapeyron.jl/xxxxx/database/SAFT/PCSAFT/data_PCSAFT_unlike.csv"

```
"""
function getpaths(location::AbstractString; relativetodatabase::Bool=false)
    # We do not use realpath here directly because we want to make the .csv suffix optional.
    filepath = relativetodatabase ? normpath(dirname(pathof(Clapeyron)), "..", "database", location) : location
    isfile(filepath) && return [realpath(filepath)]
    isfile(filepath * ".csv") && return [realpath(filepath * ".csv")]
    if !isdir(filepath)
        relativetodatabase ? error("The path ", location, " does not exist in the Clapeyron database.") :
            error("The path ", location, " does not exist.")
    end
    files = joinpath.(filepath, readdir(filepath))
    return realpath.(files[isfile.(files) .& (getfileextension.(files) .== "csv")])
end

function flattenfilepaths(locations,userlocations)
    res = vcat(
        reduce(vcat, getpaths.(locations; relativetodatabase=true), init = String[]),
        reduce(vcat, getpaths.(userlocations), init = String[]),
        String[]
    )
    return res
end

#function getparams()

"""
    params, sites = getparams(components,locations;kwargs...)

returns a `Dict{String,ClapeyronParam}` containing all the parameters found for the list of components
in the available CSVs. `locations` are the locations relative to `Clapeyron` database. the available keywords are the ones used in [`ParamOptions`](@ref)

if `return_sites` is set to false, `getparams` will only return the found params.
"""
function getparams(
        components,
        locations::Vector{String} = String[];
        userlocations::Vector{String} = String[],
        asymmetricparams::Vector{String} = String[],
        ignore_missing_singleparams::Vector{String} = String[],
        ignore_headers::Vector{String} = ["dipprnumber", "smiles"],
        verbose::Bool = false,
        species_columnreference::String = "species",
        source_columnreference::String = "source",
        site_columnreference::String = "site",
        group_columnreference::String = "groups",
        normalisecomponents::Bool = true,
        return_sites::Bool = true,
        component_delimiter::String = "~|~"
    )
    param_options = ParamOptions(
        ;
        asymmetricparams,
        ignore_missing_singleparams,
        ignore_headers,
        species_columnreference,
        source_columnreference,
        site_columnreference,
        group_columnreference,
        normalisecomponents,
        return_sites,
        component_delimiter)
    
    # locations is a list of paths relative to the Clapeyron database directory.
    # userlocations is a list of paths input by the user.
    # If parameters exist in multiple files, Clapeyron gives priority to files in later paths.
    # asymmetricparams is a list of parameters for which matrix reflection is disabled.
    # ignore_missingsingleparams gives users the option to disable component existence check in single params.                   
    return getparams(components, locations, param_options; userlocations, verbose)
end

function getparams(
        components::Vector{String},
        locations::Vector{String},
        param_options::ParamOptions;
        userlocations::Vector{String} = String[],
        verbose::Bool = false,
    )
    filepaths = flattenfilepaths(locations, userlocations)
    allcomponentsites = findsitesincsvs(components, filepaths; verbose, param_options)
    allparams, paramsourcecsvs, paramsources = createparamarrays(components, filepaths, allcomponentsites; verbose, param_options)
    result = packageparams(allparams, components, allcomponentsites, paramsourcecsvs, paramsources, param_options)
    if !param_options.return_sites
        return result
    end
    if any(x isa AssocParam for x in values(result))
        sites = buildsites(result, components, allcomponentsites, param_options)
        return result,sites
    else
        return result
    end
end

function getparams(
        groups::GroupParam,
        locations::Vector{String},
        param_options::ParamOptions;
        userlocations::Vector{String} = String[],
        verbose::Bool = false,
    )
    return getparams(groups.flattenedgroups, locations, param_options; userlocations, verbose)
end

function getparams(
        components::String,
        locations::Vector{String},
        param_options::ParamOptions;
        userlocations::Vector{String} = String[],
        verbose::Bool = false
    )
    return getparams([components], locations, param_options; userlocations, verbose)
end

function buildsites(
        result,
        components,
        allcomponentsites,
        param_options
    )
    n_sites_columns = param_options.n_sites_columns
    v = String[]
    for sitei in allcomponentsites
        append!(v,sitei)
    end
    unique!(v)
    iszero(length(v)) && return SiteParam(components)
    if !any(haskey(result, n_sites_columns[vi]) for vi in v)
        @error """No columns containing number of sites were found. Supposing zero sites.
        If your model doesn't use sites, but some input CSV folders contain Association params. consider using return_sites=false.
        If there are columns containing number of sites, then those aren't recognized. Consider passing a Dict with the mappings between the sites and the name of the column containing the number of said sites in with the n_sites_columns keyword"
        """
        assoc_csv = Set(String[])
        for x in values(result)
            if x isa AssocParam
                for csvx in x.sourcecsvs
                push!(assoc_csv,csvx)
                end
            end
        end
        assoc_csv = collect(assoc_csv)
        @error "Parsed Association CSV were:"
        for csv in assoc_csv
            println(csv)
        end
        return SiteParam(components)
    end
    
    n_sites_dict = Dict{String,SingleParam{Int}}(vi => result[n_sites_columns[vi]] for vi in v)    
    return SiteParam(n_sites_dict, allcomponentsites)
end

function packageparams(allparams::Dict, 
    components::Vector{String}, 
    allcomponentsites::Vector{Vector{String}},
    paramsourcecsvs::Dict{String,Set{String}}, 
    paramsources::Dict{String,Set{String}},
    param_options::ParamOptions = DefaultParamOptions)

    asymmetricparams = param_options.asymmetricparams
    ignore_missingsingleparams = param_options.ignore_missing_singleparams
    # Package params into their respective Structs.
    output = Dict{String,ClapeyronParam}()
    for (param, value) ∈ allparams
        output[param] = pkgparam(param, value, components, allcomponentsites, paramsourcecsvs, paramsources; param_options)
    end
    return output
end

#SingleParam
function pkgparam(param::String,
    value::Vector{<:NumberOrString},
    components::Vector{String}, 
    allcomponentsites::Vector{Vector{String}},
    paramsourcecsvs::Dict{String,Set{String}}, 
    paramsources::Dict{String,Set{String}};
    param_options::ParamOptions = DefaultParamOptions)
    newvalue, ismissingvalues = defaultmissing(value)
    if param ∉ param_options.ignore_missing_singleparams && any(ismissingvalues)
        error("Missing values exist in single parameter ", param, ": ", value, ".")
    end
    return SingleParam(param, components, newvalue, ismissingvalues, collect(paramsourcecsvs[param]), collect(paramsources[param]))
end

#PairParam
function pkgparam(param::String,
    value::Matrix{<:NumberOrString},
    components::Vector{String}, 
    allcomponentsites::Vector{Vector{String}},
    paramsourcecsvs::Dict{String,Set{String}}, 
    paramsources::Dict{String,Set{String}};
    param_options::ParamOptions = DefaultParamOptions)
    
    param ∉ param_options.asymmetricparams && mirrormatrix!(value)
    newvalue, ismissingvalues = defaultmissing(value)
    diagidx = diagind(newvalue)
    diagvalues = view(newvalue,diagidx)

    if param ∉ param_options.ignore_missing_singleparams 
        missing_diag = view(ismissingvalues,diagidx)
        type = first(missing_diag)
        if any(!=(type),missing_diag) #if all are true or all are false 
            error("Partial missing values exist in diagonal of pair parameter ", param, ": ", [value[x,x] for x ∈ 1:size(ismissingvalues,1)], ".")
       end
    end
    return PairParam(param,components, newvalue, diagvalues,ismissingvalues, collect(paramsourcecsvs[param]), collect(paramsources[param]))
end

#AssocParam
function pkgparam(param::String,
    value::Matrix{<:Matrix{<:NumberOrString}},
    components::Vector{String}, 
    allcomponentsites::Vector{Vector{String}},
    paramsourcecsvs::Dict{String,Set{String}}, 
    paramsources::Dict{String,Set{String}};
    param_options::ParamOptions = DefaultParamOptions)
    
    param ∉ param_options.asymmetricparams && mirrormatrix!(value) 
    newvalue_ismissingvalues = defaultmissing.(value)
    newvalue = first.(newvalue_ismissingvalues)
    return AssocParam(param, components, newvalue, allcomponentsites, collect(paramsourcecsvs[param]), collect(paramsources[param]))
end

function param_type(t1,t2)
    t_promoted = promote_type(t1,t2)
    local res
    if t_promoted !== Any
        res =  Union{Missing,t_promoted}
    else
        res = Union{Missing,t1,t2}
    end
    return res
end

function createparamarrays(
        components::Vector{String},
        filepaths::Vector{String},
        allcomponentsites::Vector{Vector{String}};
        verbose::Bool = false,
        param_options::ParamOptions = DefaultParamOptions
    )
    # Returns Dict with all parameters in their respective arrays.
    check_clashingheaders(filepaths, param_options)
    allparams = Dict{String,Any}()
    paramsourcecsvs = Dict{String,Set{String}}()
    paramsources = Dict{String,Set{String}}()
    # Read the filepaths in reverse in order to ensure that unused sources do not get added.
    
    for filepath ∈ reverse(filepaths)
        name, csvtype, grouptype = _readcsvtype(filepath)
        if csvtype == groupdata
            verbose && @info("Skipping groupdata csv $filepath")
            continue
        end
        headerparams = readheaderparams(filepath; param_options)
        verbose && @info("Searching for $(Symbol(csvtype)) headers $headerparams for components $components at $filepath ...")
        foundparams, paramtypes, sources = findparamsincsv(components, filepath; verbose, param_options)
        foundparams = swapdictorder(foundparams)
        for headerparam ∈ headerparams
            if !haskey(allparams, headerparam)
                if ismissing(paramtypes[headerparam])
                    allparams[headerparam] = createemptyparamsarray(missing, csvtype, components, allcomponentsites)
                else
                    allparams[headerparam] = createemptyparamsarray(paramtypes[headerparam], csvtype, components, allcomponentsites)
                end
            end
            if !haskey(paramsourcecsvs, headerparam)
                paramsourcecsvs[headerparam] = Set{String}()
                paramsources[headerparam] = Set{String}()
            end
            if csvtype == singledata
                isempty(foundparams) && continue
                for (component, value) ∈ foundparams[headerparam]
                    currenttype = nonmissingtype(eltype(allparams[headerparam]))
                    tx = param_type(paramtypes[headerparam],currenttype)
                    allparams[headerparam] = convert(Array{tx}, allparams[headerparam])
                    idx = findfirst(isequal(component), components)
                    if allparams[headerparam] isa Matrix
                        !ismissing(allparams[headerparam][idx,idx]) && continue
                        allparams[headerparam][idx,idx] = value
                    else
                        !ismissing(allparams[headerparam][idx]) && continue
                        allparams[headerparam][idx] = value
                    end
                    push!(paramsourcecsvs[headerparam], filepath)
                    !ismissing(sources[component]) && push!(paramsources[headerparam], sources[component])
                end
            end
            if csvtype == pairdata
                if allparams[headerparam] isa Vector
                    allparams[headerparam] = singletopair(allparams[headerparam], missing)
                end
                isempty(foundparams) && continue
                for (componentpair, value) ∈ foundparams[headerparam]
                    currenttype = nonmissingtype(eltype(allparams[headerparam]))
                    tx = param_type(paramtypes[headerparam],currenttype)
                    allparams[headerparam] = convert(Matrix{tx}, allparams[headerparam])
                    idx1 = findfirst(isequal(componentpair[1]), components)
                    idx2 = findfirst(isequal(componentpair[2]), components)
                    !ismissing(allparams[headerparam][idx1,idx2]) && continue
                    allparams[headerparam][idx1,idx2] = value
                    push!(paramsourcecsvs[headerparam], filepath)
                    !ismissing(sources[componentpair]) && push!(paramsources[headerparam], sources[componentpair])
                end
            end
            if csvtype == assocdata
                isempty(foundparams) && continue
                for (assocpair, value) ∈ foundparams[headerparam]
                    currenttype = nonmissingtype(eltype(first(allparams[headerparam])))
                    tx = param_type(paramtypes[headerparam],currenttype)
                    allparams[headerparam] = convert(Array{Array{tx,2},2}, allparams[headerparam])
                    idx1 = findfirst(isequal(assocpair[1][1]), components)
                    idx2 = findfirst(isequal(assocpair[1][2]), components)
                    idx21 = findfirst(isequal(assocpair[2][1]), allcomponentsites[idx1])
                    idx22 = findfirst(isequal(assocpair[2][2]), allcomponentsites[idx2])
                    !ismissing(allparams[headerparam][idx1,idx2][idx21,idx22]) && continue
                    allparams[headerparam][idx1,idx2][idx21,idx22] = value
                    push!(paramsourcecsvs[headerparam], filepath)
                    !ismissing(sources[assocpair]) && push!(paramsources[headerparam], sources[assocpair])
                end
            end
        end
    end
    return allparams, paramsourcecsvs, paramsources
end

function defaultmissing(array::Array{<:Number})
    return deepcopy(array), fill(false, size(array))
end

function defaultmissing(array::Array{<:AbstractString})
    return String.(array), fill(false, size(array))
end

function defaultmissing(array::Array{Union{Missing, T}}) where T <:AbstractString
    return String.(coalesce.(array, "")), Array(ismissing.(array))
end

function defaultmissing(array::Array{Union{Missing, T}}) where T<:Number
    default = zero(eltype(array))
    return coalesce.(array, default), Array(ismissing.(array))
end

#if an array with only missings is passed, the Resulting ClapeyronParam will be 
#of the type that this function returns
function defaultmissing(array::Array{Missing})
    n =size(array)
    return fill(0.0, n), Array(fill(true, n))
end

function defaultmissing(array::Array{Union{Missing,T1,T2}}) where {T1 <:Number,T2<:AbstractString}
    return string.(coalesce.(array,"")), Array(ismissing.(array))
end

function defaultmissing(array::Array{Any})
    return string.(coalesce.(array,"")), Array(ismissing.(array))
end

function defaultmissing(array)
    throw("Unsupported array element type $(typeof(array))")
end

function swapdictorder(dict)
    # Swap the first two levels in a nested dictionary.
    isempty(dict) && return dict
    output = Dict()
    outerkeys = keys(dict)
    try
        innerkeys = keys(first(values(dict)))
        for innerkey ∈ innerkeys, outerkey ∈ outerkeys
            if !haskey(output, innerkey)
                output[innerkey] = Dict{Any,Any}(outerkey => dict[outerkey][innerkey])
            end
            push!(output[innerkey], outerkey => dict[outerkey][innerkey])
        end
    catch e
        error("The format of the nested dictionary is not correct.")
    end
    return output
end


function col_indices(
        csvtype,
        headernames,
        param_options = DefaultParamOptions
    )
    columnreference = param_options.species_columnreference
    normalised_columnreference = normalisestring(columnreference)

    idx_species = 0
    idx_groups = 0
    idx_species1 = 0
    idx_species2 = 0
    idx_sites1 = 0
    idx_sites2 = 0

    if csvtype === singledata || csvtype === groupdata
        lookupcolumnindex = findfirst(isequal(normalised_columnreference), headernames)
        isnothing(lookupcolumnindex) && error("Header ", normalised_columnreference, " not found.")
        idx_species = lookupcolumnindex
        if csvtype === groupdata
            groupcolumnreference= param_options.group_columnreference
            normalised_groupcolumnreference = normalisestring(groupcolumnreference)
            lookupgroupcolumnindex = findfirst(isequal(normalised_groupcolumnreference), headernames)
            isnothing(lookupgroupcolumnindex) && error("Header ", normalised_groupcolumnreference, " not found.")
            idx_groups = lookupgroupcolumnindex
        end
    
    elseif csvtype === pairdata || csvtype == assocdata
        normalised_columnreference1 = normalised_columnreference * '1'
        normalised_columnreference2 = normalised_columnreference * '2'
        lookupcolumnindex1 = findfirst(isequal(normalised_columnreference1), headernames)
        lookupcolumnindex2 = findfirst(isequal(normalised_columnreference2), headernames)
        isnothing(lookupcolumnindex1) && error("Header ", normalised_columnreference1, " not found.")
        isnothing(lookupcolumnindex2) && error("Header ", normalised_columnreference2, " not found.")
        idx_species1 = lookupcolumnindex1
        idx_species2 = lookupcolumnindex2
        if csvtype == assocdata
            sitecolumnreference = param_options.site_columnreference
            normalised_sitecolumnreference = normalisestring(sitecolumnreference)
            normalised_sitecolumnreference1 = normalised_sitecolumnreference * '1'
            normalised_sitecolumnreference2 = normalised_sitecolumnreference * '2'
            lookupsitecolumnindex1 = findfirst(isequal(normalised_sitecolumnreference1), headernames)
            lookupsitecolumnindex2 = findfirst(isequal(normalised_sitecolumnreference2), headernames)
            isnothing(lookupsitecolumnindex1) && error("Header ", normalised_sitecolumnreference1, " not found.")
            isnothing(lookupsitecolumnindex2) && error("Header ", normalised_sitecolumnreference2, " not found.")
            idx_sites1 = lookupsitecolumnindex1
            idx_sites2 = lookupsitecolumnindex2
        end
    end

    _single = (idx_species,idx_groups)
    _pair = (idx_species1,idx_species2)
    _assoc = (idx_sites1,idx_sites2)
    return (_single,_pair,_assoc)
end



function findparamsincsv(components::Array{String,1},
    filepath::AbstractString;
    verbose::Bool = false,
    param_options::ParamOptions = DefaultParamOptions)

    headerparams = readheaderparams(filepath; param_options)
    sourcecolumnreference = param_options.source_columnreference
    normalisecomponents = param_options.normalisecomponents
    component_delimiter = param_options.component_delimiter

    name, csvtype, grouptype = _readcsvtype(filepath)
    df = CSV.File(filepath; header=3, pool=0, silencewarnings=true)
    csvheaders = String.(Tables.columnnames(df))
    normalised_components = normalisestring.(components, normalisecomponents)
    normalised_csvheaders = normalisestring.(csvheaders)
    normalised_headerparams = normalisestring.(headerparams)
    normalised_headerparams ⊈ normalised_csvheaders && error("Headers ", setdiff(normalised_headerparams, normalised_csvheaders), " not present in csv header.")

    foundvalues = Dict()
    paramtypes = Dict(headerparams .=> [Tables.columntype(df, Symbol(x)) for x ∈ headerparams])
    sources = Dict{Any,Union{String,Missing}}()

    normalised_sourcecolumnreference = normalisestring(sourcecolumnreference)
    getsources = false
    if normalised_sourcecolumnreference ∈ normalised_csvheaders
        getsources = true
        sourcecolumn = findfirst(isequal(normalised_sourcecolumnreference), normalised_csvheaders)
    end

    single_idx, pair_idx, assoc_idx = col_indices(csvtype, normalised_csvheaders, param_options)
    lookupcolumnindex,_ = single_idx
    lookupcolumnindex1,lookupcolumnindex2 = pair_idx
    lookupsitecolumnindex1,lookupsitecolumnindex2 = assoc_idx
    headerparams_indices = [findfirst(isequal(i),normalised_csvheaders) for i in normalised_headerparams]
    if csvtype == singledata
        for row ∈ Tables.rows(df)   
            component_split = split(row[lookupcolumnindex], component_delimiter, keepempty=false)
            for component ∈ component_split
                foundcomponentidx = findfirst(isequal(normalisestring(component,normalisecomponents)), normalised_components)
                isnothing(foundcomponentidx) && continue
                
                component = components[foundcomponentidx]
                foundvalues[component] = Dict{String,Any}()
                for (headerparam,idx) ∈ zip(headerparams,headerparams_indices)
                    foundvalues[component][headerparam] = row[idx]
                end
                verbose && @info("""Found component: $(component)
                with values: $(foundvalues[component])
                """)
                if getsources
                    source = row[sourcecolumn]
                    sources[component] = source
                else
                    sources[component] = missing
                end
            end
        end
    elseif csvtype == pairdata
        for row ∈ Tables.rows(df)
            component_split1 = split(row[lookupcolumnindex1], component_delimiter, keepempty=false)
            for component1 ∈ component_split1
                foundcomponentidx1 = findfirst(isequal(normalisestring(component1,normalisecomponents)), normalised_components)
                isnothing(foundcomponentidx1) && continue
                component_split2 = split(row[lookupcolumnindex2], component_delimiter, keepempty=false)
                for component2 ∈ component_split2
                    foundcomponentidx1 = findfirst(isequal(normalisestring(component1,normalisecomponents)), normalised_components)
                    foundcomponentidx2 = findfirst(isequal(normalisestring(component2,normalisecomponents)), normalised_components)
                    isnothing(foundcomponentidx2) && continue
                    componentpair = (components[foundcomponentidx1], components[foundcomponentidx2])
                    foundvalues[componentpair] = Dict{String,Any}()
                    for (headerparam,idx) ∈ zip(headerparams,headerparams_indices)
                        foundvalues[componentpair][headerparam] = row[idx]
                    end
                    verbose && @info("""Found component pair: ($(component1),$(component2))
                    with values: $(foundvalues[componentpair])
                    """)
                    if getsources
                        source = row[sourcecolumn]
                        sources[componentpair] = source
                    else
                        sources[componentpair] = missing
                    end
                end
            end
        end
    elseif csvtype == assocdata  
        for row ∈ Tables.rows(df)
            component_split1 = split(row[lookupcolumnindex1], component_delimiter, keepempty=false)
            for component1 ∈ component_split1
                foundcomponentidx1 = findfirst(isequal(normalisestring(component1,normalisecomponents)), normalised_components)
                isnothing(foundcomponentidx1) && continue
                component_split2 = split(row[lookupcolumnindex2], component_delimiter, keepempty=false)
                for component2 ∈ component_split2
                    foundcomponentidx2 = findfirst(isequal(normalisestring(component2,normalisecomponents)), normalised_components)
                    isnothing(foundcomponentidx2) && continue
                    site1 = row[lookupsitecolumnindex1]
                    site2 = row[lookupsitecolumnindex2]
                    assocpair = ((components[foundcomponentidx1], components[foundcomponentidx2]), (site1, site2))
                    foundvalues[assocpair] = Dict{String,Any}()
                    for (headerparam,idx) ∈ zip(headerparams,headerparams_indices)
                        foundvalues[assocpair][headerparam] = row[idx]
                    end
                    verbose && @info("""Found assoc pair: ($(component1),$(component2)) , ($(site1),$(site2))
                    with values: $(foundvalues[assocpair])
                    """)
                    if getsources
                        source = row[sourcecolumn]
                        sources[assocpair] = source
                    else
                        sources[assocpair] = missing
                    end
                end
            end
        end
    else
        error("File is of type ", String(csvtype), " and cannot be read with this function.")
    end
    return foundvalues, paramtypes, sources
end

function normalisestring(str::AbstractString, isactivated::Bool=true; tofilter::Regex=r"[ \-\_]", changecase::Bool=true)
    !isactivated && return str
    normalisedstring = replace(str, tofilter => "")
    changecase && (normalisedstring = lowercase(normalisedstring))
    return normalisedstring
end


"""
    _readcsvtype(filepath)

Reads the second line of a csv to determine parameter metadata.
The format should be

    name:{NAME},paramtype:{PARAMTYPE},grouptype:{GROUPTYPE}
"""
function _readcsvtype(filepath)::Tuple{String,CSVType,Union{Symbol,Nothing}}
    words = split(replace(rstrip(getline(String(filepath), 2), ','), " " => ""), ',')
    # Default values
    name::String = ""
    paramtype::Union{CSVType,Nothing} = nothing
    grouptype::Union{Symbol,Nothing} = nothing
    try
        for word in words
            keyvalue = split(word, ':')
            if length(keyvalue) < 2
                continue
            end
            key, value = strip.(keyvalue[1:2], ' ')
            if key == "name"
                name = value
            elseif key == "paramtype"
                paramtype = _readcsvtype_enum(lowercase(value))
            elseif key == "grouptype"
                if value != "nothing"
                    grouptype = Symbol(value)
                end
            end
        end
        if isnothing(paramtype)
            error("paramtype canot be found for csv ", filepath, ".")
        end
    catch e
        # Try parsing using legacy parser before throwing error.
        try
            return _readcsvtype_legacy(filepath)
        catch
            throw(e)
        end
    end
    return Tuple{String,CSVType,Union{Symbol,Nothing}}([name, paramtype, grouptype])
end

const _readcsvtype_keywords  = ["like", "single", "unlike", "pair", "assoc", "group", "groups"]
function _readcsvtype_legacy(filepath)
    # Old readcsvtype.
    # Searches for type from second line of CSV.
    keywords = _readcsvtype_keywords
    words = split(lowercase(rstrip(getline(String(filepath), 2), ',')), ' ')
    foundkeywords = intersect(words, keywords)
    isempty(foundkeywords) && error("Unable to determine type of database", filepath, ". Check that keyword is present on Line 2.")
    length(foundkeywords) > 1 && error("Multiple keywords found in database ", filepath, ": ", foundkeywords)
    enum = _readcsvtype_enum(only(foundkeywords))
    Base.depwarn("Old metadata format detected for ${filepath}. Metadata format for second row of CSV file has changed. Please consult documentation for details, or browse built-in database files to view the new format.", :getparams; force=true)
    Tuple{String,CSVType,Union{Symbol,Nothing}}([words[1], enum, nothing])
end

function _readcsvtype_enum(key)
    key == "single" && return singledata
    key == "like" && return singledata
    key == "pair" && return pairdata
    key == "unlike" && return pairdata
    key == "assoc" && return assocdata
    key == "group" && return groupdata
    key == "groups" && return groupdata
    error("Unable to determine database type of $key")
end

function getline(filepath::AbstractString, selectedline::Int)
    # Simple function to return text from filepath at selectedline.
    open(filepath) do file
        linecount = 1
        for line ∈ eachline(file)
            linecount == selectedline && return line
            linecount += 1
        end
        error("Selected line number exceeds number of lines in file")
    end
end

function readheaderparams(
        filepath::AbstractString;
        param_options::ParamOptions = DefaultParamOptions,
        headerline::Int = 3)
    # Returns array of filtered header strings at line 3.
    ignorelist = deepcopy(param_options.ignore_headers)
    push!(ignorelist, param_options.species_columnreference)
    push!(ignorelist, param_options.source_columnreference)
    push!(ignorelist, param_options.site_columnreference)
    headers = split(getline(filepath, headerline), ',', keepempty = false)
    while last(headers) == ""
        pop!(headers)
    end
    return String.(filter(x -> normalisestring(x; tofilter=r"[ \-\_\d]") ∉ ignorelist, headers))
end

function check_clashingheaders(
        filepaths::Vector{String},
        param_options::ParamOptions = DefaultParamOptions)
    # Raises an error if the header of any assoc parameter clashes with a non-assoc parameter.
    headerparams = String[]
    headerparams_assoc = String[]
    for filepath in filepaths
        _, csvtype, _ = _readcsvtype(filepath)
        if csvtype == singledata || csvtype == pairdata
            append!(headerparams, readheaderparams(filepath; param_options))
        elseif csvtype == assocdata
            append!(headerparams_assoc, readheaderparams(filepath; param_options))
        end
    end
    clashingheaders = intersect(headerparams, headerparams_assoc)
    !isempty(clashingheaders) && error("Headers ", clashingheaders, " appear in both loaded assoc and non-assoc files.")
end

function findsitesincsvs(
        components::Array{String,1}, 
        filepaths::Array{String,1};
        verbose::Bool = false,
        param_options::ParamOptions = DefaultParamOptions
    )
    normalisecomponents = param_options.normalisecomponents
    # Look for all relevant sites in the database.
    # Note that this might not necessarily include all sites associated with a component.
    normalised_components = normalisestring.(components, normalisecomponents)
    sites = Dict(components .=> [Set{String}() for _ ∈ 1:length(components)])
    for filepath ∈ filepaths
        name, csvtype, grouptype = _readcsvtype(filepath)
        csvtype != assocdata && continue
        df = CSV.File(filepath; header=3,silencewarnings = !verbose) 
        csvheaders = String.(Tables.columnnames(df))
        normalised_csvheaders = normalisestring.(csvheaders)
        _,pair_idx, assoc_idx = col_indices(csvtype, normalised_csvheaders, param_options)
        lookupcolumnindex1, lookupcolumnindex2 = pair_idx
        lookupsitecolumnindex1, lookupsitecolumnindex2 = assoc_idx
        for row ∈ Tables.rows(df)
            component1 = row[lookupcolumnindex1]
            component2 = row[lookupcolumnindex2]
            foundcomponentidx1 = findfirst(isequal(normalisestring(component1, normalisecomponents)), normalised_components)
            foundcomponentidx2 = findfirst(isequal(normalisestring(component2, normalisecomponents)), normalised_components)
            (isnothing(foundcomponentidx1) || isnothing(foundcomponentidx2)) && continue
            push!(sites[components[foundcomponentidx1]], row[lookupsitecolumnindex1])
            push!(sites[components[foundcomponentidx2]], row[lookupsitecolumnindex2])
        end
    end
    output = Array{Vector{String}}(undef, 0)
    for component ∈ components
        push!(output, collect(sites[component]))
    end
    verbose && @info("Found sites for $components are $(output).")
    return output
end

function findgroupsincsv(
        components::Vector{String},
        filepath::String;
        verbose::Bool = false,
        param_options::ParamOptions = DefaultParamOptions
    )
    normalisecomponents = param_options.normalisecomponents
    name, csvtype, grouptype = _readcsvtype(filepath)
    csvtype != groupdata && return Dict{String,String}()    
    normalised_components = normalisestring.(components, normalisecomponents)
    df = CSV.File(filepath; header=3, silencewarnings = !verbose)
    columns = Tables.columns(df)
    csvheaders = String.(Tables.columnnames(df))
    normalised_csvheaders = normalisestring.(csvheaders)
    single_idx, _, _ = col_indices(csvtype, normalised_csvheaders, param_options)
    lookupcolumnindex,lookupgroupcolumnindex = single_idx
    species_column = Tables.getcolumn(columns, lookupcolumnindex)
    groups_column = Tables.getcolumn(columns, lookupgroupcolumnindex)
    norm_species_column = normalisestring.(species_column, normalisecomponents)
    idx = findall(in(normalised_components), norm_species_column)
    found_comps = @view species_column[idx]
    found_groups = @view groups_column[idx]
    if verbose
        for i in 1:length(found_comps)
            @info("""Found component: $(found_comps[i]) 
            with groups: $(found_groups[i])
            """)
        end
    end
    foundgroups = Dict(comp_i => group_i for (comp_i, group_i) in zip(found_comps, found_groups))
    return foundgroups
end

function createemptyparamsarray(datatype::Type, csvtype::CSVType, components::Vector{String})
    allcomponentsites = [String[] for _ in 1:length(components)]
    return createemptyparamsarray(datatype, csvtype, components, allcomponentsites)
end

function createemptyparamsarray(datatype::Type, csvtype::CSVType, components::Vector{String}, allcomponentsites::Vector{Vector{String}})
    # Creates a missing array of the appropriate size.
    componentslength = length(components)
    csvtype == singledata && return (Array{Union{Missing,datatype}}(undef, componentslength) .= missing)
    csvtype == pairdata && return (Array{Union{Missing,datatype}}(undef, componentslength, componentslength) .= missing)
    if csvtype == assocdata
        output = Array{Array{Union{Missing,datatype},2},2}(undef, componentslength, componentslength)
        for i ∈ 1:componentslength, j ∈ 1:componentslength
            output[i,j] = (Array{Union{Missing,datatype}}(undef, length(allcomponentsites[i]), length(allcomponentsites[j])) .= missing)
        end
        return output
    end
end

function createemptyparamsarray(csvtype::CSVType, components::Vector{String})
    allcomponentsites = [String[] for _ in 1:length(components)]
    return createemptyparamsarray(missing, csvtype, components, allcomponentsites)
end

function createemptyparamsarray(csvtype::CSVType, components::Vector{String}, allcomponentsites::Vector{Vector{String}})
    return createemptyparamsarray(missing, csvtype, components, allcomponentsites)
end

_zero(t::Number) = zero(t)
_zero(::String) = ""
_zero(::Missing) = missing

function _zero(x::Type{T})  where T<:Number
    return zero(T)
end

_zero(::Type{String}) = ""
_zero(::Type{Missing}) = missing

function _zero(::Type{T}) where T <:Union{T1,Missing} where T1
    return missing
end

_iszero(t::Number) = iszero(t)
_iszero(::Missing) = true
_iszero(t::AbstractString) = isempty(t)

"""
    singletopair(params::Vector,outputmissing=zero(T))

Generates a square matrix, filled with "zeros" (considering the "zero" of a string, a empty string). 
The generated matrix will have the values of `params` in the diagonal.
If missing is passed, the matrix will be filled with `missing`
"""
function singletopair(params::Vector{T1}, ::T2 =_zero(T1)) where {T1,T2}
    len = length(params)
    T = Union{T1,T2}
    output = Matrix{T}(undef,len,len)
    fill!(output, _zero(T))
    @inbounds  for i in 1:len
        output[i,i] = params[i]
    end
    return output
end

function mirrormatrix!(matrix::Matrix{T}) where T
    # Mirrors a square matrix.
    matrixsize = size(matrix)
    matrixsize[1] != matrixsize[2] && error("Matrix is not square.")
    for i ∈ 2:matrixsize[1], j ∈ 1:i-1
        lowervalue = matrix[i,j]
        uppervalue = matrix[j,i]
        if !ismissing(lowervalue) && !ismissing(uppervalue) && lowervalue != uppervalue
            error("Dissimilar non-zero entries exist across diagonal:", lowervalue, ", ", uppervalue)
        end
        !ismissing(lowervalue) && (matrix[j,i] = lowervalue)
        !ismissing(uppervalue) && (matrix[i,j] = uppervalue)
    end
    return matrix
end

function mirrormatrix!(matrix::Array{Array{T,2},2}) where T
    # Mirrors a square matrix.
    matrixsize = size(matrix)
    matrixsize[1] != matrixsize[2] && error("Matrix is not square.")
    [mirrormatrix!(matrix[i,i]) for i ∈ 1:matrixsize[1]]
    for i ∈ 2:matrixsize[1], j ∈ 1:i-1
        matrixsize2 = size(matrix[i,j])
        for a ∈ 1:matrixsize2[1], b ∈ 1:matrixsize2[2]
            lowervalue = matrix[i,j][a,b]
            uppervalue = matrix[j,i][b,a]
            if !ismissing(lowervalue) && !ismissing(uppervalue) && lowervalue != uppervalue
                error("Dissimilar non-zero entries exist across diagonal:", lowervalue, ", ", uppervalue)
            end
            !ismissing(lowervalue) && (matrix[j,i][b,a] = lowervalue)
            !ismissing(uppervalue) && (matrix[i,j][a,b] = uppervalue)
        end
    end
    return matrix
end

function GroupParam(
        gccomponents, 
        grouplocations::Vector{String} = String[];
        groupdefinitions::Vector{GroupDefinition} = GroupDefinition[],
        usergrouplocations::Vector{String} = String[],
        verbose::Bool = false,
        param_options::ParamOptions = DefaultParamOptions
    )
    # The format for gccomponents is an arary of either the species name (if it
    # available in the Clapeyron database, or a tuple consisting of the species
    # name, followed by a list of group => multiplicity pairs.  For example:
    # gccomponents = ["ethane",
    #                ("hexane", ["CH3" => 2, "CH2" => 4]),
    #                ("octane", ["CH3" => 2, "CH2" => 6])]
    BuildSpeciesType = Union{Tuple{String, Array{Pair{String, Int64},1}}, String, Tuple{String}}
    any(.!(isa.(gccomponents, BuildSpeciesType))) && error("The format of the components is incorrect.")
    if !(gccomponents isa Union{String,Vector{String}})
        Base.depwarn("Listing group definitions in components argument is deprecated. List only component names, and then define the groups in keyword argument `groupdefinitions`, or in a csv and point to its location in `usergrouplocations`", :GroupParam; force=true)
    end
    # Groups provided explicitly in groupdefinitions have higherst priority.
    # If group not found here, look into csvs; otherwise, error.
    groupdefinitions_formatted = Dict(definition.component => Pair.([definition.groups, definition.multiplicities]) for definition ∈ groupdefinitions)
    filepaths = flattenfilepaths(grouplocations, usergrouplocations)
    componentstolookup = String[]
    #componentstolookup = filter(x-> x isa Union{String,Tuple{String}},gccomponents)
    append!(componentstolookup, [x for x ∈ gccomponents[isa.(gccomponents, String)]])
    append!(componentstolookup, [first(x) for x ∈ gccomponents[isa.(gccomponents, Tuple{String})]])
    allfoundcomponentgroups = Dict{String,String}()
    groupsourcecsvs = String[]
    for filepath in filepaths
        name, csvtype, grouptype = _readcsvtype(filepath)
        if csvtype != groupdata
            verbose && @info("Skipping $csvtype csv at $filepath")
            continue
        end
        verbose && @info("Searching for groups for components $componentstolookup at $filepath ...")
        merge!(allfoundcomponentgroups, findgroupsincsv(componentstolookup, filepath; verbose, param_options))
        append!(groupsourcecsvs, [filepath])
    end
    gccomponents_parsed = PARSED_GROUP_VECTOR_TYPE(undef,length(gccomponents))
    for (i, gccomponent) ∈ pairs(gccomponents)
        if gccomponent isa Tuple{String, Vector{Pair{String, Int64}}}
            gccomponents_parsed[i] = gccomponent
        else
            if gccomponent isa String
                component = gccomponent
            elseif gccomponent isa Tuple{String}
                component = first(gccomponent)
            end
            if haskey(groupdefinitions_formatted, component)
                gccomponents_parsed[i] = (component, groupdefinitions_formatted[component])
            elseif haskey(allfoundcomponentgroups, component)
                groupsandngroups = eval(Meta.parse(allfoundcomponentgroups[component]))
                gccomponents_parsed[i] = (component, groupsandngroups)
            else
                error("Predefined component ", component, " not found in any group definitions or input csvs.")
            end
        end
    end
    return GroupParam(gccomponents_parsed, groupsourcecsvs)
end

export getparams
