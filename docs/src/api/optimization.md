# Optimization & Bayesian

This page documents functions for parameter estimation and Bayesian inference.

## Optimization

```@docs
Filthy.DSL.optimize_ssm
Filthy.DSL.optimize_ssm_with_stderr
```

## Parameter Transformations

```@docs
Filthy.DSL.build_transformation
Filthy.DSL.transform_to_constrained
Filthy.DSL.transform_to_unconstrained
```

## Log-Density Interface

```@docs
Filthy.DSL.SSMLogDensity
Filthy.DSL.logdensity
```

## Prior Distributions

```@docs
Filthy.DSL.FlatPrior
Filthy.DSL.NormalPrior
Filthy.DSL.InverseGammaPrior
Filthy.DSL.CompositePrior
```
