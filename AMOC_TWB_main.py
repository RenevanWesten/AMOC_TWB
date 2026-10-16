#Main to perform the AMOC feedback analysis

from pylab import *
import numpy as np
import glob, os
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy import stats
import statsmodels.api as sm
from scipy.stats import linregress
from matplotlib import rc, rcParams
from AMOC_TWB_functions import *

def MovingAverage(time_year, data_year, moving_average = 25):
	#Moving average of yearly-averaged time series
	
	time_2		= np.zeros(len(time_year) - moving_average + 1)
	data_2		= ma.masked_all((len(time_2)))

	for time_i in range(len(time_2)):
		time_2[time_i]		= time_year[time_i] + (moving_average-1) / 2.0
		data_2[time_i]		= np.mean(data_year[time_i:time_i+moving_average], axis = 0)
		
	return time_2, data_2	

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

depth_max_index	= 33    #This is custom, check your actual AMOC

fh = netcdf.Dataset('Atlantic_sector_zonal_averages.nc', 'r')

time_all	= fh.variables['time'][:]  
depth		= fh.variables['depth'][:depth_max_index] 
dz		    = fh.variables['dz'][:depth_max_index] 
lat		    = fh.variables['lat'][:]   
temp_all	= fh.variables['TEMP'][:, :depth_max_index]    		
salt_all	= fh.variables['SALT'][:, :depth_max_index]   		
dens_all	= fh.variables['RHO'][:, :depth_max_index]  
pot_dens_all= fh.variables['PD'][:, :depth_max_index]  
lat_volume	= fh.variables['lat_volume'][:depth_max_index]  		

fh.close()

#AMOC strength at 1000 m depth
fh = netcdf.Dataset('AMOC_transport_depth_0-1000m.nc', 'r')	

AMOC		= fh.variables['Transport'][:]	#AMOC strength (Sv)

fh.close()

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#Initialise the model with the dimensions for the southern and northern box
AMOC_TWB	= AMOC_TWB(time_all, lat, -58, -34, 40, 65, depth, dz, pot_dens_all, lat_volume)

#Update the AMOC scaling coefficient using the actual AMOC strength of the first 50 model years (or period of choice)
AMOC_TWB.AMOC_SCALING_update(AMOC, 0, 50)

#Determine the contributions from temperature and salinity changes
AMOC_TWB.ALPHA_BETA_update(temp_all, salt_all, dens_all)

AMOC_TWB.PlotVerticalProfiles()


#-----------------------------------------------------------------------------------------
#----------------------------------------PLOTTING-----------------------------------------
#-----------------------------------------------------------------------------------------
#--------------------THE FIGURES BELOW ARE SPECIFICALLY FOR THE HR-POP--------------------
#-----------------------------------------------------------------------------------------

depth_crop					                   = 1000
factor_depth_crop				               = 4

depth_plot					                   = np.copy(depth)
depth_TWB_plot					               = AMOC_TWB.depth_TWB

depth_plot[depth_plot > depth_crop] 		   = ((depth_plot[depth_plot > depth_crop] - depth_crop) / factor_depth_crop) + depth_crop
depth_TWB_plot[depth_TWB_plot > depth_crop]   = ((depth_TWB_plot[depth_TWB_plot > depth_crop] - depth_crop) / factor_depth_crop) + depth_crop


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

fh = netcdf.Dataset('AMOC_structure_year_1-50.nc', 'r')

depth_stream	= fh.variables['depth'][:]		
lat_stream	    = fh.variables['lat'][:]
lat_index	    = (np.abs(lat_stream - 26)).argmin()	
AMOC_z		    = np.mean(fh.variables['AMOC'][:, :, lat_index], axis = 0)	#AMOC strength (Sv)

fh.close()

depth_stream[depth_stream > depth_crop] 		= ((depth_stream[depth_stream > depth_crop] - depth_crop) / factor_depth_crop) + depth_crop

fig, ax	= subplots()

graph_1	= ax.plot(AMOC_z, depth_stream, '-', color = 'goldenrod', linewidth = 2.0, label = '$\Psi(26^{\circ}$N$, z)$')
graph_2	= ax.plot(np.mean(AMOC_TWB.AMOC_TWB_z[0:50], axis = 0), depth_TWB_plot, '-k', linewidth = 2.0, label = r'$\hat{\Psi}(z)$')

ax.set_xlim(-5, 25)
ax.set_ylim(((5500 - depth_crop) / factor_depth_crop) + depth_crop, -25)
ax.set_xlabel('Streamfunction (Sv)')
ax.set_ylabel('Depth (m)')
ax.grid()

labels_rescaled =  ax.get_yticks()
for label_i in range(len(labels_rescaled)):
	if labels_rescaled[label_i] > depth_crop:
		#Rescale the xlabels
		labels_rescaled[label_i]	= ((labels_rescaled[label_i] - depth_crop) * factor_depth_crop) + depth_crop
	
labels_rescaled	= labels_rescaled.astype(int)

ax.set_yticklabels(labels_rescaled)	

graphs	      	= graph_1 + graph_2
legend_labels 	= [l.get_label() for l in graphs]
rc('text', usetex=True)
legend_1	= ax.legend(graphs, legend_labels, loc='lower right', ncol=1, framealpha = 1.0, numpoints = 1, fontsize = 13)
rc('text', usetex=False)
		
ax.set_title('AMOC and reconstruction (model year 1 - 50)')

show()

