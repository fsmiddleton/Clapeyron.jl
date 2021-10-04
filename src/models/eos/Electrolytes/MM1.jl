abstract type MM1Model <: RSPModel end

struct MM1Param <: EoSParam
    gamma::PairParam{Float64}
    theta::PairParam{Float64}
    coordz::PairParam{Float64}
    mu::SingleParam{Float64}
    polarizability::SingleParam{Float64}
end

struct MM1{A<:SAFTModel} <: MM1Model
    components::Array{String,1}
    solvents::Array{String,1}
    ions::Array{String,1}
    icomponents::UnitRange{Int}
    isolvents::UnitRange{Int}
    iions::UnitRange{Int}
    params::MM1Param
    assocmodel::A
    absolutetolerance::Float64
    references::Array{String,1}
end

@registermodel MM1
export MM1
function MM1(solvents,salts; assocmodel=sCPA, userlocations::Vector{String}=String[],assoc_userlocations::Vector{String}=String[], verbose::Bool=false)
    ion_groups = GroupParam(salts, ["Electrolytes/salts.csv"]; verbose=verbose)

    ions = ion_groups.flattenedgroups
    components = deepcopy(solvents)
    append!(components,ions)
    icomponents = 1:length(components)
    isolvents = 1:length(solvents)
    iions = (length(solvents)+1):length(components)

    params = getparams(components, ["Electrolytes/MM1_like.csv","Electrolytes/MM1_unlike.csv"]; userlocations=userlocations, verbose=verbose)
    params["gamma"].values .*= pi/180
    gamma = params["gamma"]
    params["theta"].values .*= pi/180
    theta = params["theta"]
    coordz = params["coordz"]
    params["mu"].values .*= 1. /(299792458)*1e-21
    mu = params["mu"]
    params["polarizability"].values .*= 1e-40
    polarizability = params["polarizability"]
    packagedparams = MM1Param(gamma,theta,coordz,mu,polarizability)

    init_assocmodel = assocmodel(components;userlocations=assoc_userlocations)

    references = [""]
    
    model = MM1(components, solvents, ions, icomponents, isolvents, iions, packagedparams, init_assocmodel, 1e-12,references)
    return model
end

function RSP(model::MM1Model, V, T, z)
    _1 = one(V+T+first(z))

    μ0 = model.params.mu.values
    γ = model.params.gamma.values
    θ = model.params.theta.values
    α = model.params.polarizability.values
    z̄ = model.params.coordz.values

    sites = model.assocmodel.sites.i_sites

    x = z ./ sum(z)
    ρ = N_A*sum(z)/(V)

    A = ρ/(3*ϵ_0)*sum(x[i]*α[i] for i ∈ @comps)
    ϵ_inf = (2*A+1)/(1-A)

    X_ = X(model.assocmodel,V,T,z)

    P = [[_1 for i ∈ model.isolvents] for j ∈ model.isolvents]
    for i ∈ model.isolvents
        for j ∈ model.isolvents
            P[i][j] = ρ/N_A*x[j]*sum(sum(Δ(model.assocmodel,V,T,z,i,j,a,b)*X_[i][a]*X_[j][b] for a ∈ sites[i]) for b ∈ sites[j])
        end
    end

    g = [_1 for i ∈ model.isolvents]

    for i ∈ model.isolvents
        g[i] = 1+sum(z̄[i,j].*P[i][j].*cos(γ[i,j])./(sum(P[i][j] for j ∈ 1:length(model.solvents)).*cos(θ[i,j])+1).*μ0[j]/μ0[i] for j ∈ 1:length(model.solvents))
    end


    B = ρ/(9*ϵ_0*k_B*T)*sum(x[i]*g[i]*μ0[i]^2 for i ∈ 1:length(model.solvents))

    poly = [2,-(ϵ_inf+(ϵ_inf+2)^2*B),-ϵ_inf^2]

    return (-poly[2]+sqrt(poly[2]^2-4*poly[1]*poly[3]))/(2*poly[1])
end

is_splittable(::MM1) = false