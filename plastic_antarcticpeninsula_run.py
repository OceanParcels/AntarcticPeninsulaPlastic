from parcels import FieldSet, Field, ParticleSet, JITParticle, ErrorCode, AdvectionRK4, BrownianMotion2D
from datetime import timedelta as delta
from argparse import ArgumentParser
from glob import glob
import numpy as np


def set_hycom_grid(files):
    filenames = {'U': files, 'V': files}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'Latitude', 'lon': 'Longitude', 'time': 'MT', 'depth': 'Depth'}
    indices = {'lat': range(1, 1500), 'depth': [0]}
    for v in ['MaskUvel', 'MaskVvel']:
        filenames[v] = "/Users/erik/Codes/ParcelsRuns/Hydrodynamics_AuxiliaryFiles/Hycom/boundary_velocities.nc"
        variables[v] = v

    fset = FieldSet.from_netcdf(filenames, variables, dimensions, indices=indices)
    fset.add_periodic_halo(zonal=True, halosize=10)
    fset.MaskUvel.units = fset.U.units
    fset.MaskVvel.units = fset.V.units
    return fset


def set_ww3_grid(stokesfiles, fset):

    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    uuss = Field.from_netcdf(stokesfiles, 'uuss', dimensions)
    vuss = Field.from_netcdf(stokesfiles, 'vuss', dimensions)
    uuss.add_periodic_halo(zonal=True, meridional=False, halosize=3)
    vuss.add_periodic_halo(zonal=True, meridional=False, halosize=3)
    uuss.units = fset.U.units
    vuss.units = fset.V.units
    return uuss, vuss


def WrapLon(particle, fieldset, time, dt):
    if particle.lon > fieldset.halo_east:
        particle.lon = particle.lon - 360.
    if particle.lon < fieldset.halo_west:
        particle.lon = particle.lon + 360.


def AdvectionRK4_stokes(particle, fieldset, time, dt):
    u1 = fieldset.U[time, particle.lon, particle.lat, particle.depth] + fieldset.uuss[time, particle.lon, particle.lat, particle.depth]
    v1 = fieldset.V[time, particle.lon, particle.lat, particle.depth] + fieldset.vuss[time, particle.lon, particle.lat, particle.depth]

    lon1, lat1 = (particle.lon + u1 * .5 * dt, particle.lat + v1 * .5 * dt)
    u2 = fieldset.U[time + .5 * dt, lon1, lat1, particle.depth] + fieldset.uuss[time + .5 * dt, lon1, lat1, particle.depth]
    v2 = fieldset.V[time + .5 * dt, lon1, lat1, particle.depth] + fieldset.vuss[time + .5 * dt, lon1, lat1, particle.depth]

    lon2, lat2 = (particle.lon + u2 * .5 * dt, particle.lat + v2 * .5 * dt)
    u3 = fieldset.U[time + .5 * dt, lon2, lat2, particle.depth] + fieldset.uuss[time + .5 * dt, lon2, lat2, particle.depth]
    v3 = fieldset.V[time + .5 * dt, lon2, lat2, particle.depth] + fieldset.vuss[time + .5 * dt, lon2, lat2, particle.depth]

    lon3, lat3 = (particle.lon + u3 * dt, particle.lat + v3 * dt)
    u4 = fieldset.U[time + dt, lon3, lat3, particle.depth] + fieldset.uuss[time + dt, lon3, lat3, particle.depth]
    v4 = fieldset.V[time + dt, lon3, lat3, particle.depth] + fieldset.vuss[time + dt, lon3, lat3, particle.depth]

    particle.lon += (u1 + 2 * u2 + 2 * u3 + u4) / 6. * dt
    particle.lat += (v1 + 2 * v2 + 2 * v3 + v4) / 6. * dt


def BoundaryVels(particle, fieldset, time, dt):
    bvu = fieldset.MaskUvel[0, particle.lon, particle.lat, particle.depth]
    bvv = fieldset.MaskVvel[0, particle.lon, particle.lat, particle.depth]
    particle.lon += bvu * dt  # taking Euler step only
    particle.lat += bvv * dt


def OutOfBounds(particle, fieldset, time, dt):
    particle.delete()


def run_hycom_particles(lons, lats, locs, rundays, files, stokesfiles):
    print(locs)
    fset = set_hycom_grid(files)
    uuss, vuss = set_ww3_grid(stokesfiles, fset)
    fset.add_field(uuss)
    fset.add_field(vuss)

    size2D = (fset.U.grid.ydim, fset.U.grid.xdim)
    fset.add_field(Field('Kh_zonal', data=10*np.ones(size2D), lon=fset.U.grid.lon, lat=fset.U.grid.lat, mesh='spherical', allow_time_extrapolation=True))
    fset.add_field(Field('Kh_meridional', data=10*np.ones(size2D), lon=fset.U.grid.lon, lat=fset.U.grid.lat, mesh='spherical', allow_time_extrapolation=True))

    nperloc = 100
    pset = ParticleSet.from_list(fieldset=fset, pclass=JITParticle, lon=np.tile(lons, [nperloc]),
                                 lat=np.tile(lats, [nperloc]))

    ofile = pset.ParticleFile(name='antarcticplastic_wstokes_5yr_%02d.nc' % locs[0], outputdt=delta(days=1))

    kernels = pset.Kernel(AdvectionRK4) + BrownianMotion2D + BoundaryVels + WrapLon
    pset.execute(kernels, runtime=delta(days=rundays), dt=-delta(minutes=5),
                 output_file=ofile, recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('-l', '--loc', type=int, default=0)
    args = p.parse_args()

    lats = [-64.989, -64.619, -64.767, -64.528, -63.907, -63.395, -63.156, -63.452, -62.636, -62.298, -61.417, -62.249]
    lons = [-63.383, -63.172, -62.658, -64.329, -64.138, -61.633, -60.631, -59.440, -58.998, -57.524, -55.251, -57.036]
    lons = [l + 360 for l in lons]
    locs = [      9,       8,       7,      10,      11,      12,       6,       5,       4,       3,       1,       2]
    days = [     14,      16,      16,      18,      18,      19,      19,      19,      21,      23,      25,      26]  # all in Feb 2017

    if args.loc > 0:
        to_run = [i for i, lc in enumerate(locs) if lc == args.loc]
    else:
        to_run = range(0, 12)

    ddir = '/Volumes/data01/HYCOMdata/GLBa0.08_expt90_surf/hycom_GLBu0.08_912_'
    # ddir = '/Users/erik/Desktop/HycomAntarctic/hycom_GLBu0.08_912_'

    stokesfiles=sorted(glob('/Volumes/data01/WaveWatch3data/WW3*'))
    allfiles = sorted(glob(ddir + "201*00_t000.nc"))
    rundays = 365 * 5

    for i in to_run:
        ifiles = [j for j, s in enumerate(allfiles) if "201702%02d" % days[i] in s]
        files = allfiles[ifiles[0] - rundays - 5:ifiles[0]]
        run_hycom_particles([lons[i]], [lats[i]], [locs[i]], rundays, files, stokesfiles)
