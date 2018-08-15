from parcels import FieldSet, Field, ParticleSet, JITParticle, ErrorCode, AdvectionRK4, BrownianMotion2D, Variable
import datetime
from datetime import timedelta as delta
from glob import glob
import numpy as np


def set_fields(hycomfiles, stokesfiles):
    dimensions = {'lat': 'Latitude', 'lon': 'Longitude', 'time': 'MT', 'depth': 'Depth'}
    bvelfile = '/Users/erik/Codes/ParcelsRuns/Hydrodynamics_AuxiliaryFiles/Hycom/boundary_velocities.nc'
    MaskUvel = Field.from_netcdf(bvelfile, 'MaskUvel', dimensions)
    MaskVvel = Field.from_netcdf(bvelfile, 'MaskVvel', dimensions)

    uhycom = Field.from_netcdf(hycomfiles, 'u', dimensions, fieldtype='U')
    vhycom = Field.from_netcdf(hycomfiles, 'v', dimensions, fieldtype='V', grid=uhycom.grid, timeFiles=uhycom.timeFiles)

    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    uuss = Field.from_netcdf(stokesfiles, 'uuss', dimensions, fieldtype='U')
    vuss = Field.from_netcdf(stokesfiles, 'vuss', dimensions, fieldtype='V', grid=uuss.grid, timeFiles=uuss.timeFiles)

    fieldset = FieldSet(U=[uhycom, uuss], V=[vhycom, vuss])
    fieldset.add_field(MaskUvel)
    fieldset.add_field(MaskVvel)
    fieldset.MaskUvel.units = fieldset.U[0].units
    fieldset.MaskVvel.units = fieldset.V[0].units

    fieldset.add_periodic_halo(zonal=True, meridional=False, halosize=5)
    return fieldset


def WrapLon(particle, fieldset, time, dt):
    if particle.lon > 360.:
        particle.lon = particle.lon - 360.
    if particle.lon < 0.:
        particle.lon = particle.lon + 360.


def BoundaryVels(particle, fieldset, time, dt):
    bvu = fieldset.MaskUvel[0, particle.lon, particle.lat, particle.depth]
    bvv = fieldset.MaskVvel[0, particle.lon, particle.lat, particle.depth]
    particle.lon += bvu * dt  # taking Euler step only
    particle.lat += bvv * dt


def OutOfBounds(particle, fieldset, time, dt):
    particle.delete()


ddir = '/Volumes/data01/HYCOMdata/GLBa0.08_expt90_surf/hycom_GLBu0.08_912_'
# ddir = '/Users/erik/Desktop/HycomAntarctic/hycom_GLBu0.08_912_'

datestr = '201*'
stokesfiles = sorted(glob('/Volumes/data01/WaveWatch3data/WW3-GLOB-30M_%s' % datestr))
allfiles = sorted(glob(ddir + datestr + '00_t000.nc'))
rundays = 365 * 7

fset = set_fields(allfiles, stokesfiles)

size2D = (fset.U[0].grid.ydim, fset.U[0].grid.xdim)
fset.add_field(Field('Kh_zonal', data=10*np.ones(size2D), lon=fset.U[0].grid.lon, lat=fset.U[0].grid.lat, mesh='spherical', allow_time_extrapolation=True))
fset.add_field(Field('Kh_meridional', data=10*np.ones(size2D), lon=fset.U[0].grid.lon, lat=fset.U[0].grid.lat, mesh='spherical', allow_time_extrapolation=True))

lats = [-64.989, -64.619, -64.767, -64.528, -63.907, -63.395, -63.156, -63.452, -62.636, -62.298, -61.417, -62.249]
lons = [-63.383, -63.172, -62.658, -64.329, -64.138, -61.633, -60.631, -59.440, -58.998, -57.524, -55.251, -57.036]
lons = [l + 360 for l in lons]
locs = [9, 8, 7, 10, 11, 12, 6, 5, 4, 3, 1, 2]
days = [14, 16, 16, 18, 18, 19, 19, 19, 21, 23, 25, 26]  # all in Feb 2017

dates = [datetime.datetime(2017, 2, d) for d in days]
nperloc = 100


class AntarcticParticle(JITParticle):
    loc = Variable('loc', dtype=np.float32, initial=0.)


pset = ParticleSet(fieldset=fset, pclass=AntarcticParticle, lon=np.repeat(lons, [nperloc]),
                   lat=np.repeat(lats, [nperloc]), time=np.repeat(dates, [nperloc]),
                   loc=np.repeat(locs, [nperloc]))

ofile = pset.ParticleFile(name='antarcticplastic_wstokes_7yr.nc', outputdt=delta(days=1))

kernels = pset.Kernel(AdvectionRK4) + BrownianMotion2D + BoundaryVels + WrapLon
pset.execute(kernels, runtime=delta(days=rundays), dt=-delta(hours=1),
             output_file=ofile, recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})
