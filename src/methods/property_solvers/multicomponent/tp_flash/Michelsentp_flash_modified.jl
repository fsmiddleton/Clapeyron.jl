

function modified_dgibbs_obj!(model::EoSModel, p, T, z, phasex, phasey, ny_var,
                               vcache, nx, ny, in_equilibria, non_inx, non_iny;
                               F=nothing, G=nothing, H=nothing)
    # Objetive Function to minimize the Gibbs Free Energy
    # It computes the Gibbs free energy, its gradient and its hessian

    ny[in_equilibria] = ny_var
    nx[in_equilibria] = z[in_equilibria] .- ny[in_equilibria]
    # nx = z .- ny

    nxsum = sum(nx)
    nysum = sum(ny)
    x = nx ./ nxsum
    y = ny ./ nysum

    # Volumes are set from local cache to reuse their values for following
    # Iterations
    volx,voly = vcache[:]

    if H !== nothing
        # Computing Gibbs Energy Hessian
        lnϕx, ∂lnϕ∂nx, ∂lnϕ∂Px, volx = ∂lnϕ∂n∂P(model, p, T, x; phase=phasex, vol0=volx)
        lnϕy, ∂lnϕ∂ny, ∂lnϕ∂Py, voly = ∂lnϕ∂n∂P(model, p, T, y; phase=phasey, vol0=voly)

        ∂ϕx = ∂lnϕ∂nx[in_equilibria, in_equilibria]
        ∂ϕy = ∂lnϕ∂ny[in_equilibria, in_equilibria]

        ∂ϕx .-= 1
        ∂ϕy .-= 1
        ∂ϕx ./= nxsum
        ∂ϕy ./= nysum
        for (i,idiag) in pairs(diagind(∂ϕy))
            ∂ϕx[idiag] += 1/nx[i]
            ∂ϕy[idiag] += 1/ny[i]
        end

        #∂ϕx = eye./nx .- 1/nxsum .+ ∂lnϕ∂nx/nxsum
        #∂ϕy = eye./ny .- 1/nysum .+ ∂lnϕ∂ny/nysum
        H .= ∂ϕx .+ ∂ϕy
    else
        lnϕx, volx = lnϕ(model, p, T, x; phase=phasex, vol0=volx)
        lnϕy, voly = lnϕ(model, p, T, y; phase=phasey, vol0=voly)
    end
    #volumes are stored in the local cache
    vcache[:] .= (volx,voly)

    ϕx = log.(x) .+ lnϕx
    ϕy = log.(y) .+ lnϕy

    # to avoid NaN in Gibbs energy
    ϕx[non_inx] .= 0.
    ϕy[non_iny] .= 0.

    if G !== nothing
        # Computing Gibbs Energy gradient
        G .= (ϕy .- ϕx)[in_equilibria]
    end

    if F != nothing
        # Computing Gibbs Energy
        FO = dot(ny,ϕy) + dot(nx,ϕx)
        return FO
    end

end


function modified_gibbs_obj!(model::EoSModel, p, T, z, phasex, phasey, ny_var,
                               vcache, nx, ny, in_equilibria, non_inx, non_iny;
                               F=nothing, G=nothing)
    # Objetive Function to minimize the Gibbs Free Energy
    # It computes the Gibbs free energy, its gradient and its hessian

    ny[in_equilibria] = ny_var
    nx[in_equilibria] = z[in_equilibria] .- ny[in_equilibria]
    # nx = z .- ny

    nxsum = sum(nx)
    nysum = sum(ny)
    x = nx ./ nxsum
    y = ny ./ nysum

    # Volumes are set from local cache to reuse their values for following
    # Iterations
    volx,voly = vcache[:]

    lnϕx, volx = lnϕ(model, p, T, x; phase=phasex, vol0=volx)
    lnϕy, voly = lnϕ(model, p, T, y; phase=phasey, vol0=voly)

    #volumes are stored in the local cache
    vcache[:] .= (volx,voly)

    ϕx = log.(x) .+ lnϕx
    ϕy = log.(y) .+ lnϕy

    # to avoid NaN in Gibbs energy
    ϕx[non_inx] .= 0.
    ϕy[non_iny] .= 0.

    if G !== nothing
        # Computing Gibbs Energy gradient
        G .= (ϕy .- ϕx)[in_equilibria]
    end

    if F != nothing
        # Computing Gibbs Energy
        FO = dot(ny,ϕy) + dot(nx,ϕx)
        return FO
    end

end


function tp_flash_michelsen_modified(model::EoSModel, p, T, z; equilibrium=:vle, K0=nothing,
                                     x0=nothing, y0=nothing, vol0=(nothing, nothing),
                                     K_tol=1e-16, itss=10, second_order=false,
                                     non_inx_list=[], non_iny_list=[], reduced=false)


    if !reduced
        model_full,z_full = model,z
        model,z_nonzero = index_reduction(model_full,z_full)
        z = z_full[z_nonzero]
    end

    if is_vle(equilibrium)
        phasex = :liquid
        phasey = :vapor
    elseif is_lle(equilibrium)
        phasex = :liquid
        phasey = :liquid
    end

    # Setting the initial guesses for volumes
    vol0 === nothing && (vol0 = (nothing,nothing))
    volx, voly = vol0


    nc = length(model)

    # constructing non-in-x list
    if !isnothing(non_inx_list)
        non_inx_names_list = [x for x in non_inx_list if x in model.components]
    else
        non_inx_names_list = String[]
    end

    if !isnothing(non_iny_list)
        non_iny_names_list = [x for x in non_iny_list if x in model.components]
    else
        non_iny_names_list = String[]
    end

    # constructing non-in-x list
    non_inx = Bool.(zeros(nc))
    # constructing non-in-y list
    non_iny = Bool.(zeros(nc))

    for i in 1:nc
        component = model.components[i]
        if component in non_inx_names_list
            non_inx[i] = true
        end

        if component in non_iny_names_list
            non_iny[i] = true
        end
    end

    inx = .!non_inx
    iny = .!non_iny

    """
    # old api when using indices instead of components names
    non_inx = Bool.(zeros(nc))
    non_inx[non_inx_list] .= 1
    inx = .!non_inx

    # constructing non-in-y list
    non_iny = Bool.(zeros(nc))
    non_iny[non_iny_list] .= 1
    iny = .!non_iny
    """

    # components that are allowed to be in two phases
    # in_equilibria = inx .&& iny
    in_equilibria = inx .* iny

    # Computing the initial guess for the K vector
    if ~isnothing(K0)
        K = 1. * K0
        lnK = log.(K)
    elseif ~isnothing(x0) && ~isnothing(y0)
        x = x0
        y = y0
        lnϕx, volx = lnϕ(model, p, T, x; phase=phasex, vol0=volx)
        lnϕy, voly = lnϕ(model, p, T, y; phase=phasey, vol0=voly)
        lnK = lnϕx - lnϕy
        K = exp.(lnK)
    elseif is_vle(equilibrium)
        # Wilson Correlation for K
        K = wilson_k_values(model,p,T)
        lnK = log.(K)
    else
        err() = @error("""You need to provide either an initial guess for the partion constant K
                        or for compositions of x and y for LLE""")
        err()
    end

    _1 = one(p+T+first(z))
    # Initial guess for phase split
    βmin = max(0., minimum(((K.*z .- 1) ./ (K .-  1.))[K .> 1]))
    βmax = min(1., maximum(((1 .- z) ./ (1. .- K))[K .< 1]))
    β = _1*(βmin + βmax)/2

    # Stage 1: Successive Substitution
    singlephase = false
    error_lnK = _1
    it = 0

    x = similar(z)
    y = similar(z)

    while error_lnK > K_tol && it < itss
        it += 1
        lnK_old = lnK .* _1
        error_β = _1
        it_rr = 0
        while error_β > 1e-8 && it_rr < 10
            it_rr += 1
            FOi = (K .- 1) ./ (1. .+ β .* (K .- 1))

            # modification for non-in-y components Ki -> 0
            FOi[non_iny] .= - 1. / (1. - β)
            # modification for non-in-x components Ki -> ∞
            FOi[non_inx] .= 1. / β

            OF = dot(z, FOi)
            dOF = - dot(z, FOi.^2)
            d2OF = 2. *dot(z, FOi.^3)

            dβ = - (2*OF*dOF)/(2*dOF^2-OF*d2OF)
            β = β + dβ
            error_β = abs(dβ)
            # println(it_rr, " ", β, " ", dβ, " ", OF)
        end
    singlephase = !(0 <= β <= 1)

    x .= z ./ (1. .+ β .* (K .- 1))
    y .= x .* K

    # modification for non-in-y components Ki -> 0
    x[non_iny] = z[non_iny] / (1. - β)
    y[non_iny] .= 0.

    # modification for non-in-x components Ki -> ∞
    x[non_inx] .= 0.
    y[non_inx] .= z[non_inx] / β

    x ./= sum(x)
    y ./= sum(y)

    # Updating K's
    lnϕx, volx = lnϕ(model, p, T, x; phase=phasex, vol0=volx)
    lnϕy, voly = lnϕ(model, p, T, y; phase=phasey, vol0=voly)
    lnK .= lnϕx .- lnϕy
    K .= exp.(lnK)

    # Computing error
    # error_lnK = sum((lnK .- lnK_old).^2)
    error_lnK = dnorm(lnK,lnK_old,1)
    end

    # Stage 2: Minimization of Gibbs Free Energy
    if error_lnK > K_tol && it == itss &&  ~singlephase
        # println("Second order minimization")
        nx = zeros(nc)
        ny = zeros(nc)

        ny[non_iny] .= 0.
        ny[non_inx] = z[non_inx]

        nx[non_iny] = z[non_iny]
        nx[non_inx] .= 0.

        vcache = [volx, voly]
        ny_var0 = y[in_equilibria] * β

        if second_order
            dfgibbs!(F, G, H, ny_var) = modified_dgibbs_obj!(model, p, T, z, phasex, phasey, ny_var, vcache,
                                                 nx, ny, in_equilibria, non_inx, non_iny;
                                                 F=F, G=G, H=H)

            sol = Solvers.optimize(Solvers.only_fgh!(dfgibbs!), ny_var0, Solvers.LineSearch(Solvers.Newton()))
        else

            fgibbs!(F, G, H, ny_var) = modified_gibbs_obj!(model, p, T, z, phasex, phasey, ny_var, vcache,
                                                           nx, ny, in_equilibria, non_inx, non_iny;
                                                           F=F, G=G)

            sol = Solvers.optimize(Solvers.only_fgh!(fgibbs!), ny_var0, Solvers.LineSearch(Solvers.Newton()))


        end
        ny_var = Solvers.x_sol(sol)

        ny[in_equilibria] = ny_var
        nx[in_equilibria] = z[in_equilibria] .- ny[in_equilibria]

        nxsum = sum(nx)
        nysum = sum(ny)
        x = nx ./ nxsum
        y = ny ./ nysum
        β = sum(ny)
    end

    if singlephase
        β = zero(β)/zero(β)
        # Gustavo: the fill! function was giving an error
        # fill!(x,z)
        # fill!(y,z)
        x .= z
        y .= z
    end

    if !reduced
        x = index_expansion(x,z_nonzero)
        y = index_expansion(y,z_nonzero)
    end

    return x, y, β

end

# export tp_flash_michelsen_modified
export ModifiedMichelsenTPFlash
