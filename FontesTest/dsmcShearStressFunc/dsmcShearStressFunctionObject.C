/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2024 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "functionObject.H"
#include "fvMesh.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"
#include "vectorIOField.H"
#include "labelIOList.H"
#include "IOdictionary.H"
#include <fstream>
#include <string>
#include <cstdio>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class dsmcShearStressFunctionObject
:
    public functionObject
{
    //- Reference to mesh
    const fvMesh& mesh_;

public:

    TypeName("dsmcShearStress");

    dsmcShearStressFunctionObject
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~dsmcShearStressFunctionObject() = default;

    virtual bool read(const dictionary& dict)
    {
        return true;
    }

    virtual bool execute()
    {
        return write();
    }

    virtual bool write();

    virtual bool clear()
    {
        return true;
    }
};


defineTypeNameAndDebug(dsmcShearStressFunctionObject, 0);
addToRunTimeSelectionTable
(
    functionObject,
    dsmcShearStressFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

dsmcShearStressFunctionObject::dsmcShearStressFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObject(name),
    mesh_(refCast<const fvMesh>(runTime.lookupObject<objectRegistry>("region0")))
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * //

bool dsmcShearStressFunctionObject::write()
{
    // Create/access the shear stress field
    if (!mesh_.objectRegistry::found("shearStressDSMC"))
    {
        volSymmTensorField* fieldPtr = new volSymmTensorField
        (
            IOobject
            (
                "shearStressDSMC",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedSymmTensor("zero", dimPressure, symmTensor::zero)
        );
        
        mesh_.objectRegistry::store(fieldPtr);
    }

    volSymmTensorField& sigma = 
        const_cast<volSymmTensorField&>
        (
            mesh_.objectRegistry::lookupObject<volSymmTensorField>
            ("shearStressDSMC")
        );

    // Reset field to zero
    sigma = dimensionedSymmTensor("zero", dimPressure, symmTensor::zero);

    // Try to read lagrangian particle fields from disk
    const fileName lagrangianPath = 
        mesh_.time().timePath() / "lagrangian" / "dsmc";

    const fileName UPath = lagrangianPath / "U";
    const fileName cellIdPath = lagrangianPath / "cellId";

    if (!isFile(UPath))
    {
        WarningInFunction
            << "Particle velocity field not found: " << UPath << nl;
        sigma.write();
        return true;
    }

    // Read particle velocity field
    vectorIOField U_particles
    (
        IOobject
        (
            "U",
            mesh_.time().timeName(),
            "lagrangian/dsmc",
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    if (U_particles.empty())
    {
        InfoInFunction << "No particles in cloud." << nl;
        sigma.write();
        return true;
    }

    // Try to read cell IDs (map particles to cells)
    // If cellId doesn't exist, parse lagrangian files that are stored as
    // Cloud<dsmcParcel> (positions/coordinates), not as plain vectorField.
    labelList cellIds(U_particles.size(), -1);
    label mappedCount = 0;
    
    if (isFile(cellIdPath))
    {
        // Read explicit cellId if available
        labelIOList cellIdField
        (
            IOobject
            (
                "cellId",
                mesh_.time().timeName(),
                "lagrangian/dsmc",
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        cellIds = cellIdField;
        forAll(cellIds, i)
        {
            if (cellIds[i] >= 0 && cellIds[i] < mesh_.nCells())
            {
                ++mappedCount;
            }
        }
        InfoInFunction << "Read cellId field from file." << nl;
    }
    else
    {
        fileName positionsPath = lagrangianPath / "positions";
        fileName coordPath = lagrangianPath / "coordinates";

        // First choice: positions file has "(x y z) cellId" per line
        if (isFile(positionsPath))
        {
            std::ifstream posFile(positionsPath.c_str());

            if (posFile.good())
            {
                std::string line;
                label pI = 0;

                while (std::getline(posFile, line) && pI < cellIds.size())
                {
                    int cI = -1;

                    // Parse first integer that appears after closing ')'
                    if (std::sscanf(line.c_str(), " (%*[^)]) %d", &cI) == 1)
                    {
                        cellIds[pI] = cI;
                        if (cI >= 0 && cI < mesh_.nCells())
                        {
                            ++mappedCount;
                        }
                        ++pI;
                    }
                }

                InfoInFunction
                    << "Parsed cell IDs from positions file. Mapped "
                    << mappedCount << " of " << cellIds.size() << " particles." << nl;
            }
            else
            {
                WarningInFunction
                    << "Cannot open positions file: " << positionsPath << nl;
            }
        }

        // Fallback: coordinates file lines are like
        // "(tetFrac...) cellId faceId tetPtI" for Cloud<dsmcParcel>
        else if (isFile(coordPath))
        {
            std::ifstream coordFile(coordPath.c_str());

            if (coordFile.good())
            {
                std::string line;
                label pI = 0;

                while (std::getline(coordFile, line) && pI < cellIds.size())
                {
                    int cI = -1;

                    // coordinates line format:
                    // "(tetFractions...) cellId faceId tetPtI"
                    if (std::sscanf(line.c_str(), " (%*[^)]) %d", &cI) == 1)
                    {
                        cellIds[pI] = cI;
                        if (cI >= 0 && cI < mesh_.nCells())
                        {
                            ++mappedCount;
                        }
                        ++pI;
                    }
                }

                InfoInFunction
                    << "Parsed cell IDs from coordinates file. Mapped "
                    << mappedCount << " of " << cellIds.size() << " particles." << nl;
            }
            else
            {
                WarningInFunction
                    << "Cannot open coordinates file: " << coordPath << nl;
            }
        }
        else
        {
            WarningInFunction
                << "No cellId, positions, or coordinates field found." << nl;
        }
    }

    InfoInFunction
        << "Mapped " << mappedCount << " / " << U_particles.size()
        << " particles to valid cells." << nl;


    // Get particle mass from constant/dsmcProperties
    scalar m = 1.0;
    
    IOdictionary dsmcDict
    (
        IOobject
        (
            "dsmcProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    if (dsmcDict.found("m"))
    {
        m = dsmcDict.get<scalar>("m");
    }
    else if (dsmcDict.found("typeIdList") && dsmcDict.found("moleculeProperties"))
    {
        const wordList typeIds(dsmcDict.lookup("typeIdList"));
        const dictionary& molProps = dsmcDict.subDict("moleculeProperties");

        if (typeIds.size() && molProps.found(typeIds[0]))
        {
            const dictionary& speciesDict = molProps.subDict(typeIds[0]);
            if (speciesDict.found("mass"))
            {
                m = speciesDict.get<scalar>("mass");
            }
        }
    }

    scalar nEquivalentParticles = 1.0;
    if (dsmcDict.found("nEquivalentParticles"))
    {
        nEquivalentParticles = dsmcDict.get<scalar>("nEquivalentParticles");
    }

    InfoInFunction << "Particle mass m = " << m << " kg" << nl;
    InfoInFunction << "nEquivalentParticles = " << nEquivalentParticles << nl;

    Info<< "Processing " << U_particles.size() << " particles" << nl;

    // Step 1: Count particles per cell and compute mean velocity
    scalarField nParticles(mesh_.nCells(), 0.0);
    vectorField U_mean(mesh_.nCells(), vector::zero);

    forAll(U_particles, i)
    {
        label cellI = cellIds[i];

        if (cellI >= 0 && cellI < mesh_.nCells())
        {
            U_mean[cellI] += U_particles[i];
            nParticles[cellI] += 1.0;
        }
    }

    // Normalize mean velocities
    forAll(U_mean, cellI)
    {
        if (nParticles[cellI] > 0.5)
        {
            U_mean[cellI] /= nParticles[cellI];
        }
    }

    if (!mappedCount)
    {
        WarningInFunction
            << "No particles mapped to mesh cells. Writing zero shearStressDSMC." << nl;
        sigma.write();
        return true;
    }

    // Step 2: Compute stress tensor from velocity deviations
    forAll(U_particles, i)
    {
        label cellI = cellIds[i];

        if (cellI >= 0 && cellI < mesh_.nCells() && nParticles[cellI] > 0.5)
        {
            // Velocity deviation from mean
            vector dc = U_particles[i] - U_mean[cellI];

            // Accumulate raw second moment sum: (u-uMean) ⊗ (u-uMean)
            symmTensor sigma_contrib
            (
                dc.x() * dc.x(),
                dc.x() * dc.y(),
                dc.x() * dc.z(),
                dc.y() * dc.y(),
                dc.y() * dc.z(),
                dc.z() * dc.z()
            );

            // Accumulate
            sigma[cellI] += sigma_contrib;
        }
    }

    // Convert second moment sum to stress tensor with DSMC parcel weighting:
    // sigma = (m * nEquivalentParticles / Vcell) * sum_p[(u-uMean)⊗(u-uMean)]
    const scalarField& V = mesh_.V();
    forAll(sigma, cellI)
    {
        if (nParticles[cellI] > 0.5 && V[cellI] > SMALL)
        {
            sigma[cellI] *= (m * nEquivalentParticles / V[cellI]);
        }
    }

    // Populate boundary values for output visualization/post-processing.
    // The field is cell-based, so for calculated wall patches we copy
    // neighbouring internal values to the patch field.
    forAll(sigma.boundaryField(), patchI)
    {
        if (sigma.boundaryField()[patchI].type() == "calculated")
        {
            sigma.boundaryFieldRef()[patchI] = sigma.boundaryField()[patchI].patchInternalField();
        }
    }

    sigma.write();

    Info<< "dsmcShearStress: shearStressDSMC computed and written at t = " 
        << mesh_.time().timeName() << nl;

    return true;
}


} // End namespace Foam

// ************************************************************************* //



