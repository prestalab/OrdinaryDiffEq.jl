# OrdinaryDiffEqCore

Core infrastructure for OrdinaryDiffEq.jl solvers.

## Floating-time steps

In the native `solve!` and `step!` paths, timestep-changeable methods with finite
floating clocks use the representable accepted time increment. Preparation may shorten
a proposal to respect its bound, `dtmax`, and the next time stop; it never
lengthens a step solely to snap the clock to an event.

Distinct nearby stops are integrated rather than skipped. If no representable
progress is possible under the bounds, integration fails without reporting a
clock-only accepted step. Existing nonlinear, local-error, domain and adaptive
minimum-step guards remain active.

Accepted-step callbacks and current-step interpolation observe the accepted
interval, separately from the controller's next-step proposal.
