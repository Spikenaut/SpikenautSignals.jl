# gbm_surprise.jl — Geometric Brownian Motion Surprise Z-Score
#
# Computes the last log-return's deviation from the local GBM drift/volatility
# estimate.  Positive values → upward surprise; negative → downward surprise.
# Useful for anomaly detection and adaptive SNN threshold modulation.
#
# Extracted from Eagle-Lander market_sde.jl (Spikenaut-Capital).

"""
    compute_gbm_surprise(prices::AbstractVector{<:Real}) -> Float64

Compute the GBM surprise Z-score of the last log-return in `prices`.

The surprise is `(last_return − drift) / volatility`, where `drift` and
`volatility` are estimated from the full `prices` series.

Returns a value in `[-3.0, 3.0]`.  Returns `0.0` when the series has fewer
than 5 elements or volatility is effectively zero.

# Example
```julia
using SpikenautSignals
prices = [100.0, 101.0, 100.5, 102.0, 103.5]
z = compute_gbm_surprise(prices)   # positive → last move was above expectation
```
"""
function compute_gbm_surprise(prices::AbstractVector{<:Real})::Float64
    length(prices) < 5 && return 0.0

    fp      = Float64.(prices)
    returns = log.(fp[2:end] ./ fp[1:end-1])
    drift   = mean(returns)
    vol     = std(returns)

    vol < 1e-10 && return 0.0

    surprise = (returns[end] - drift) / vol
    return clamp(surprise, -3.0, 3.0)
end
