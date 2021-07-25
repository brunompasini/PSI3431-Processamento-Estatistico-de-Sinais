using LinearAlgebra, FixedPointNumbers
# Vítor H. Nascimento, abril 2020
# Para o curso PSI3431-Processamento Estatístico de Sinais

"""
    fxfilt(b::Vector,
            a::Vector,
            x::Vector)

Filters signal `x` by an IIR filter with numerator coefficients in vector `b/a[1]`, and denominator coefficients in vector `a/a[1]`, using the Direct I canonical form.
"""
function fxfilt(b::Vector,a::Vector,x::Vector)

    N = length(a)
    anorm = a[2:end]/a[1]
    bnorm = b / a[1]
    M = length(b)
    Nx = length(x)
    Nmax = max(N,M)
    y = zeros(Nx+Nmax-1)
    xext = [zeros(Nmax-1);x]
    for n = 1:Nx
        y[n+Nmax-1] = -anorm⋅y[n+Nmax-2:-1:n+Nmax-N] + bnorm⋅xext[n+Nmax-1:-1:n+Nmax-M]
    end
    return y[Nmax:end]
end

"""
    fxfilt(b::Vector,
            a::Vector,
            x::Vector{Fixed{Q,C}})

Filters quantized signal `x` by an IIR filter with numerator coefficients in vector `b/a[1]`, and denominator coefficients in vector `a/a[1]`, using the Direct I canonical form.
    The coefficients are of type `Vector`, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  Intermediate operations are single-precision.
"""
function fxfilt(b::Vector,a::Vector,x::Vector{Fixed{Q,C}}) where {Q <: Signed, B, C}

    N = length(a)
    anorm = Float64.(a[2:end])/a[1]
    bnorm = Float64.(b) / a[1]
    M = length(b)
    Nmax = max(N-1,M)
    Nmin = min(N-1,M)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx+Nmax-1))
    xext = [Fixed{Q,C}.(zeros(Nmax-1));x]
    for n = 1:Nx
        for i=1:Nmin
            y[n+Nmax-1] += Fixed{Q,C}(-anorm[i]*y[n+Nmax-1-i]) + Fixed{Q,C}(bnorm[i]*xext[n+Nmax-i])
        end
        if M < N-1
            for i=Nmin+1:N-1
                y[n+Nmax-1] -= Fixed{Q,C}(anorm[i]*y[n+Nmax-1-i])
            end
        elseif N-1 < M
            for i=Nmin+1:M
                y[n+Nmax-1] += Fixed{Q,C}(bnorm[i]*xext[n+Nmax-i])
            end
        end
    end
    return y[Nmax:end]
end

"""
    fxfilt(b::Vector,
            a::Vector,
            x::Vector{Fixed{Q,C}},
            precdupla::Bool)

Filters quantized signal `x` by an IIR filter with numerator coefficients in vector `b/a[1]`, and denominator coefficients in vector `a/a[1]`, using the Direct I canonical form.
    The coefficients are of type `Vector`, the input is `Fixed{Q,C}` and the output is `Fixed{Q,C}`.  intermediate operations are double-precision if precdupla=true.
"""
function fxfilt(b::Vector,a::Vector,x::Vector{Fixed{Q,C}}, precdupla::Bool) where {Q <: Signed, B, C}

    if !precdupla
        return fxfilt(b,a,x)
    end
    N = length(a)
    anorm = (Float64.(a[2:end])/Float64(a[1]))
    bnorm = (Float64.(b) / Float64(a[1]))
    M = length(b)
    Nmax = max(N-1,M)
    Nmin = min(N-1,M)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx+Nmax-1))
    xext = [Fixed{Q,C}.(zeros(Nmax-1));x]
    for n = 1:Nx
        res = Fixed{Int128,2C}(0.0)
        for i=1:Nmin
            res += Fixed{Int128,2C}(-anorm[i]*y[n+Nmax-1-i] + bnorm[i]*xext[n+Nmax-i])
        end
        if M < N-1
            for i=Nmin+1:N-1
                res -= Fixed{Int128,2C}(anorm[i]*y[n+Nmax-1-i])
            end
        elseif N-1 < M
            for i=Nmin+1:M
                res += Fixed{Int128,2C}(bnorm[i]*xext[n+Nmax-i])
            end
        end
        y[n+Nmax-1] = Fixed{Q,C}(res)
    end
    return y[Nmax:end]
end


"""
    fxfilt(b::Vector{Fixed{T,B}},
            a::Vector{Fixed{T,B}},
            x::Vector{Fixed{Q,C}})

Filters quantized signal `x` by an IIR filter with quantized numerator coefficients in vector `b/a[1]`, and quantized denominator coefficients in vector `a/a[1]`, using the Direct I canonical form.
    The coefficients are of type `Fixed{T,B}`, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are single-precision.
"""
function fxfilt(b::Vector{Fixed{T,B}},a::Vector{Fixed{T,B}},x::Vector{Fixed{Q,C}}) where {T <: Signed, Q <: Signed, B, C}

    N = length(a)
    anorm = Fixed{T,B}.(Float64.(a[2:end])/a[1])
    bnorm = Fixed{T,B}.(Float64.(b) / a[1])
    M = length(b)
    Nmax = max(N-1,M)
    Nmin = min(N-1,M)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx+Nmax-1))
    xext = [Fixed{Q,C}.(zeros(Nmax-1));x]
    for n = 1:Nx
        for i=1:Nmin
            y[n+Nmax-1] += Fixed{Q,C}(-anorm[i]*y[n+Nmax-1-i]) + Fixed{Q,C}(bnorm[i]*xext[n+Nmax-i])
        end
        if M < N-1
            for i=Nmin+1:N-1
                y[n+Nmax-1] -= Fixed{Q,C}(anorm[i]*y[n+Nmax-1-i])
            end
        elseif N-1 < M
            for i=Nmin+1:M
                y[n+Nmax-1] += Fixed{Q,C}(bnorm[i]*xext[n+Nmax-i])
            end
        end
    end
    return y[Nmax:end]
end

"""
    fxfilt(b::Vector{Fixed{T,B}},
            a::Vector{Fixed{T,B}},
            x::Vector{Fixed{Q,C}},
            precdupla::Bool)

Filters quantized signal `x` by an IIR filter with quantized numerator coefficients in vector `b/a[1]`, and quantized denominator coefficients in vector `a/a[1]`, using the Direct I canonical form.
    The coefficients are of type `Fixed{T,B}`, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are double-precision if precdupla=true.
"""
function fxfilt(b::Vector{Fixed{T,B}},a::Vector{Fixed{T,B}},x::Vector{Fixed{Q,C}}, precdupla::Bool) where {T <: Signed, Q <: Signed, B, C}

    if !precdupla
        return fxfilt(b,a,x)
    end
    N = length(a)
    anorm = Fixed{Int128,2C}.(Float64.(a[2:end])/Float64(a[1]))
    bnorm = Fixed{Int128,2C}.(Float64.(b) / Float64(a[1]))
    M = length(b)
    Nmax = max(N-1,M)
    Nmin = min(N-1,M)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx+Nmax-1))
    xext = [Fixed{Q,C}.(zeros(Nmax-1));x]
    for n = 1:Nx
        res = Fixed{Int128,2C}(0.0)
        for i=1:Nmin
            res += -anorm[i]*y[n+Nmax-1-i] + bnorm[i]*xext[n+Nmax-i]
        end
        if M < N-1
            for i=Nmin+1:N-1
                res -= anorm[i]*y[n+Nmax-1-i]
            end
        elseif N-1 < M
            for i=Nmin+1:M
                res += bnorm[i]*xext[n+Nmax-i]
            end
        end
        y[n+Nmax-1] = Fixed{Q,C}(res)
    end
    return y[Nmax:end]
end
"""
    fxfilt(b::Vector,
            a::Number,
            x::Vector)

Filters signal `x` by an FIR filter with numerator coefficients in vector `b/a`.
"""
function fxfilt(b::Vector,a::Number,x::Vector)
    bnorm = b / a
    M = length(b)
    Nx = length(x)
    y = zeros(Nx)
    xext = [zeros(M-1);x]
    for n = 1:Nx
        y[n] = bnorm⋅xext[n+M-1:-1:n]
    end
    return y
end

"""
    fxfilt(b::Vector,
            x::Vector)

Filters signal `x` by an FIR filter with numerator coefficients in vector `b`.
"""
function fxfilt(b::Vector,x::Vector)

    M = length(b)
    Nx = length(x)
    y = zeros(Nx)
    xext = [zeros(M-1);x]
    for n = 1:Nx
        y[n] = b⋅xext[n+M-1:-1:n]
    end
    return y
end


"""
    fxfilt(b::Vector{Fixed{T,B}},
            x::Vector{Fixed{Q,C}},
            precdupla::Bool)

Filters quantized signal `x` by an FIR filter with quantized numerator coefficients in vector `b`.  The coefficients are of type `Fixed{T,B}`, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are single-precision.
"""
function fxfilt(b::Vector{Fixed{T,B}},x::Vector{Fixed{Q,C}}) where {T<:Signed, Q<:Signed, B, C}

    M = length(b)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx))
    xext = [Fixed{Q,C}.(zeros(M-1));x]
    for n = 1:Nx
        for i=1:M
            y[n] += Fixed{Q,C}(b[i]*xext[n+M-i])
        #y[n] = Fixed{Q,C}(b⋅xext[n+M-1:-1:n]) # Usar isto varia o resultado
        end
    end
    return y
end
"""
    fxfilt(b::Vector{Fixed{T,B}},
            x::Vector{Fixed{Q,C}},
            precdupla::Bool)

Filters quantized signal `x` by an FIR filter with quantized numerator coefficients in vector `b`.  The coefficients are of type `Fixed{T,B}`, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are double-precision (2C), if precdupla=true.
"""
function fxfilt(b::Vector{Fixed{T,B}},x::Vector{Fixed{Q,C}},precdupla::Bool) where {T<:Signed, Q<:Signed, B, C}

    M = length(b)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx))
    xext = [Fixed{Q,C}.(zeros(M-1));x]
    if !precdupla
        return fxfilt(b,x)
    end
    for n = 1:Nx
        res = Fixed{Int128,2C}(0.0)
        for i=1:M
            res += (Fixed{Int128,2C}(b[i])*Fixed{Int128,2C}(xext[n+M-i]))
        end
        #y[n] = Fixed{Q,C}(b⋅xext[n+M-1:-1:n]) # Usar isto varia o resultado
        y[n] = Fixed{Q,C}(res)
    end
    return y
end

"""
    fxfilt(b::Vector,
            x::Vector{Fixed{Q,C}})

Filters quantized signal `x` by an FIR filter with float numerator coefficients in vector `b`.  The coefficients are of any type, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are single-precision (C).
"""
function fxfilt(b::Vector,x::Vector{Fixed{Q,C}}) where {Q<:Signed, C}

    M = length(b)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx))
    xext = [Fixed{Q,C}.(zeros(M-1));x]
    for n = 1:Nx
        for i=1:M
            y[n] += Fixed{Q,C}(b[i]*xext[n+M-i])
        end
    end
    return y
end

"""
    fxfilt(b::Vector,
            x::Vector{Fixed{Q,C}},
            precdupla::Bool)

Filters quantized signal `x` by an FIR filter with float numerator coefficients in vector `b`.  The coefficients are of any type, the input is `Fixed{Q,C}`, and the output is `Fixed{Q,C}`.  intermediate operations are double-precision (2C), if precdupla=true.
"""
function fxfilt(b::Vector,x::Vector{Fixed{Q,C}},precdupla::Bool) where {Q<:Signed, C}
    if !precdupla
        return fxfilt(b,x)
    end
    M = length(b)
    Nx = length(x)
    y = Fixed{Q,C}.(zeros(Nx))
    xext = [Fixed{Q,C}.(zeros(M-1));x]
    for n = 1:Nx
        res = Fixed{Int128,2C}(0.0)
        for i=1:M
            res += Fixed{128,2C}(b[i])*Fixed{128,2C}(xext[n+M-i])
        end
        y[n] = Fixed{Q,C}(res)
    end
    return y
end

"""
    fxfilt(b::Vector{Fixed{T,B}},
            x::Vector{Q})

Filters signal `x` by an FIR filter with quantized numerator coefficients in vector `b`.  The coefficients are of type `Fixed{T,B}`, the input is `Number`, and the output is `Float64` or `Complex{Float64}`, depending on the type of x.
"""
function fxfilt(b::Vector{Fixed{T,B}},x::Vector{Q}) where {T<:Signed, Q, B}

    M = length(b)
    Nx = length(x)
    if Q <: Complex
        y = Complex{Float64}.(zeros(Nx))
        xext = Complex{Float64}.([zeros(M-1);x])
    else
        y = zeros(Nx)
        xext = [zeros(M-1);Float64.(x)]
    end
    for n = 1:Nx
        y[n] = (b⋅xext[n+M-1:-1:n])
    end
    return y
end
