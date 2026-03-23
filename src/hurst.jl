# hurst.jl — Hurst Exponent via R/S Analysis
#
# Computes trend conviction / long-range dependence of a price series.
# H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
#
# Extracted from Eagle-Lander market_fractal.jl (Spikenaut-Capital).

"""
    compute_hurst(prices::AbstractVector{<:Real}) -> Float64

Estimate the Hurst exponent of `prices` using simplified R/S analysis.

Returns a value in `[0.2, 0.8]`.  Returns `0.5` (random walk) when the
series is too short (< 20 elements) or degenerate (zero variance).

# Example
```julia
using SpikenautSignals
prices = cumsum(randn(200)) .+ 100.0
h = compute_hurst(prices)   # typically near 0.5 for a random walk
```
"""
function compute_hurst(prices::AbstractVector{<:Real})::Float64
    length(prices) < 20 && return 0.5

    diffs = diff(Float64.(prices))
    m     = mean(diffs)
    s     = std(diffs)

    s < 1e-10 && return 0.5   # degenerate (constant series)

    cum_dev = cumsum(diffs .- m)
    r = maximum(cum_dev) - minimum(cum_dev)

    h = log(r / s) / log(length(prices))
    return clamp(h, 0.2, 0.8)
end
