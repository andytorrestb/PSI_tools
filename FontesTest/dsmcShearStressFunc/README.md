# DSMC Shear Stress Function Object

Runtime function object for calculating the DSMC shear stress tensor using kinetic theory.

## Build Instructions

From the case directory (`FontesTest/`), compile the function object library:

```bash
cd dsmcShearStressFunc
wmake libso
```


## Theory

The shear stress tensor is computed from individual particle velocities using kinetic theory:

$$\sigma_{ij} = \frac{m}{N_p} \sum_{k=1}^{N_p} (\mathbf{u}_k - \mathbf{U}_{mean,i}) (\mathbf{u}_k - \mathbf{U}_{mean,j})$$

where:
- $m$ is the particle mass (from `constant/dsmcProperties`)
- $\mathbf{u}_k$ is the velocity of particle $k$
- $\mathbf{U}_{mean}$ is the mean velocity per cell
- $N_p$ is the number of particles in the cell

## Configuration

In `system/controlDict`, add:

```
functions
{
    shearStressTensor
    {
        type            dsmcShearStress;
        libs            (libdsmcShearStressFunctionObject);
        executeControl  writeTime;
        writeControl    writeTime;
    }
}
```

## Output

The function object creates a `volSymmTensorField` named `shearStressDSMC` written to each time directory.

## Requirements

- OpenFOAM v2512 with dsmcFoam solver
- Molecular mass `m` defined in `constant/dsmcProperties`

## Implementation Details

The function object uses the generic `Cloud<particle>` API to iterate over particles without requiring solver-specific headers. A two-pass algorithm:

1. **First pass**: Count particles and compute mean velocity per cell
2. **Second pass**: Calculate stress tensor contributions from velocity deviations
3. **Normalization**: Divide by particle count per cell

The resulting field has dimensions of pressure (stress) and represents the kinetic contribution to the momentum transport tensor.


