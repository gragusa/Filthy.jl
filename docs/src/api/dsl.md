# DSL & Templates

This page documents the domain-specific language for model specification and pre-built templates.

## Core Types

```@docs
Filthy.DSL.SSMParameter
Filthy.DSL.SSMSpec
Filthy.DSL.FixedValue
Filthy.DSL.ParameterRef
```

## Model Introspection

```@docs
Filthy.DSL.param_names
Filthy.DSL.n_params
Filthy.DSL.initial_values
Filthy.DSL.param_bounds
```

## Model Building

```@docs
Filthy.DSL.build_linear_state_space
Filthy.DSL.ssm_loglik
Filthy.DSL.objective_function
```

## Pre-Built Templates

### Local Level Model

```@docs
Filthy.DSL.local_level
```

### Local Linear Trend

```@docs
Filthy.DSL.local_linear_trend
```

### AR(1) Model

```@docs
Filthy.DSL.ar1
```

### ARMA Model

```@docs
Filthy.DSL.arma
```

### Dynamic Factor Model

```@docs
Filthy.DSL.dynamic_factor
```

## Custom Model Specification

```@docs
Filthy.DSL.custom_ssm
Filthy.DSL.FreeParam
```

## Parameter Expressions

```@docs
Filthy.DSL.ParamExpr
Filthy.DSL.MatrixExpr
```

## DNS/Svensson Yield Curve Helpers

```@docs
Filthy.DSL.build_dns_loadings
Filthy.DSL.build_svensson_loadings
Filthy.DSL.dns_loading1
Filthy.DSL.dns_loading2
```
