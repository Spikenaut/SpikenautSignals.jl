# hawkes.jl — Self-Exciting Hawkes Intensity Proxy
#
# Computes a momentum / adrenaline signal proportional to recent jump activity.
# Higher intensity → more self-exciting price movements → higher SNN arousal.
#
# Extracted from Eagle-Lander market_hawkes.jl (Spikenaut-Capital).

"""
    compute_hawkes(prices::AbstractVector{<:Real}) -> Float64

Estimate the Hawkes self-exciting intensity of `prices` from recent absolute
price differences.

Returns a value in `[0.5, 3.0]`.  Returns `1.0` (neutral) when the series
has fewer than 2 elements.

# Example
```julia
using SpikenautSignals
prices = [100.0, 100.5, 102.0, 101.0, 103.5]
λ = compute_hawkes(prices)   # > 1.0 for volatile series
```
"""
function compute_hawkes(prices::AbstractVector{<:Real})::Float64
    length(prices) < 2 && return 1.0

    diffs     = abs.(diff(Float64.(prices)))
    intensity = 1.0 + mean(diffs) * 10.0
    return clamp(intensity, 0.5, 3.0)
end
