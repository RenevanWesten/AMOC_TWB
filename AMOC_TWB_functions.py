#Class for reconstructing the AMOC in thermal wind balance (TWB).
#Procedure is documented in Vanderborgt et al. (2025, https://doi.org/10.1175/JCLI-D-24-0570.1).
#Feedback Processes Causing an AMOC Collapse in the Community Earth System Model

from pylab import *
import numpy
import math
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy import stats
import statsmodels.api as sm
from scipy.stats import linregress
import gsw
from scipy.interpolate import CubicSpline
from matplotlib.patches import Patch

class AMOC_TWB:
    """Model input should be the zonally-averaged quantities in (depth x lat) dimensions.
    Normalisation field is optional, otherwise is based on the latitude weight and vertical grid spacing"""

    def __init__(self, time, lat, south_lat_1, south_lat_2, north_lat_1, north_lat_2, depth, dz, PD, lat_volume = None):
        """Initialise the TWB reconstruction, which includes the coordinates, boundaries of the boxes and the potential density field"""
        
        self.time               = time              #General time array (e.g., model years)
        self.lat                = lat               #Latitude coordinate
        self.south_lat_1        = south_lat_1       #Southern boundary of southern box
        self.south_lat_2        = south_lat_2       #Northern boundary of southern box
        self.north_lat_1        = north_lat_1       #Southern boundary of northern box
        self.north_lat_2        = north_lat_2       #Northern boundary of northern box  
        self.depth              = depth             #Depth coordinate
        self.dz                 = dz                #Layer thickness of depth coordinate
        self.PD                 = PD                #Potential density (time, depth, lat)
        self.AMOC_scale         = 1.0               #Scaling parameter, advise to update when AMOC strength is provided
        self.scale_salt_temp    = -0.5              #Scaling parameter, advice to update when temperature and salinity profiled are provided
        self.salt_scale         = -0.3              #Scaling parameter between upper and lower limb, advise to update when salinity profiles are provided
        self.alpha              = 1.3 * 10**(-4.0)  #Thermal expansion coefficient, advise to update when temperature profiles are provided
        self.beta               = 7.6 * 10**(-4.0)  #Haline contraction coefficient, advise to update when salinity profiles are provided
        self.g                  = 9.81              #Gravitional acceleration
        self.f_cor              = 1 * 10**(-4.0)    #Coriolis
        self.rho_0              = 1028.             #Reference density
        self.S_0                = 35.0              #Reference salinity
        
        if lat_volume is not None:
            #Use the provided normalisation
            self.lat_volume = lat_volume
            
        else:
            print("Estimate the summed volume for every depth x lat coordinate""")
            lat_volume  = ma.masked_all((len(self.depth), len(self.lat)))
            
            for depth_i in range(len(self.depth)):
                #Get the lat weights times the vertical spacing
                lat_volume[depth_i] = np.cos(lat * np.pi / 180) * dz[depth_i]

            self.lat_volume = lat_volume
        
        #Determine the potential densities over the southern and northern box   
        self.PD_south, self.PD_south_VA, self.PD_south_tilde        = self.BOXES(self.PD, self.south_lat_1, self.south_lat_2)
        self.PD_north, self.PD_north_VA, self.PD_north_tilde        = self.BOXES(self.PD, self.north_lat_1, self.north_lat_2)

        #Reconstruct the AMOC from the meridional potential density difference
        self.depth_TWB, AMOC_TWB_z, AMOC_TWB_1000   = self.PD_diff_to_AMOC(self.PD_south[0], self.PD_north[0])
            
        AMOC_TWB_z      = ma.masked_all((len(self.PD), len(self.depth_TWB)))
        AMOC_TWB_1000       = ma.masked_all(len(self.PD))
        AMOC_TWB_1000_VA    = ma.masked_all(len(self.PD))
        AMOC_TWB_1000_tilde = ma.masked_all(len(self.PD))
        
        for time_i in range(len(self.PD)):
            #The total TWB
            depth_TWB, AMOC_TWB_z[time_i], AMOC_TWB_1000[time_i]    = self.PD_diff_to_AMOC(self.PD_south[time_i], self.PD_north[time_i])
            depth_TWB, AMOC_TWB_VA_z, AMOC_TWB_1000_VA[time_i]  = self.PD_diff_to_AMOC(self.PD_south_VA[time_i], self.PD_north_VA[time_i])
            depth_TWB, AMOC_TWB_tilde_z, AMOC_TWB_1000_tilde[time_i]= self.PD_diff_to_AMOC(self.PD_south_tilde[time_i], self.PD_north_tilde[time_i])
            
        #Reconstructions and the scaling depth coefficient at 1000 m depth
        self.depth_TWB      = depth_TWB
        self.AMOC_TWB_z     = AMOC_TWB_z
        self.AMOC_TWB_1000  = AMOC_TWB_1000
        self.AMOC_TWB_1000_VA   = AMOC_TWB_1000_VA
        self.AMOC_TWB_1000_tilde= AMOC_TWB_1000_tilde
        self.D_scale        = np.sqrt(0.5 * (1000 * self.depth_TWB[-1] - 1000**2.0))
    
    def PD_diff_to_AMOC(self, PD_south, PD_north):
        """Computes a vertical profile of the overturning streamfunction in the North Atlantic 
        based on Nikurashin and Vallis, (2012) and Jansen et al., (2018) with boundary 
        conditions psi = 0 at the surface and bottom of the ocean (or z_NADW)."""

        #Remove all masked elements (usually restricted by the northern box)
        depth_index = np.where((PD_south.mask == False) & (PD_north.mask == False))[0]
    
        depth       = self.depth[depth_index]
        dz          = self.dz[depth_index]
        PD_south    = PD_south[depth_index]
        PD_north    = PD_north[depth_index]

        #Get the layer thickness to determine AMOC at exactly z = 1000 m
        dz_1000     = np.copy(dz)

        for depth_i in range(1, len(depth)):
            if np.sum(dz_1000[:depth_i]) >= 1000:
                dz_1000[depth_i-1]  = dz_1000[depth_i-1] - (np.sum(dz_1000[:depth_i]) - 1000)
                dz_1000[depth_i:]   = 0.0
    
        #Now get new depth coordinate for the AMOC strength, starting at z = 0
        depth_AMOC      = np.zeros(len(depth)+1)
        depth_AMOC_1000 = np.zeros(len(depth)+1)

        for depth_i in range(1, len(depth_AMOC)):
            #Add each layer thickness for the new depth coordinate
            depth_AMOC[depth_i]     = depth_AMOC[depth_i-1] + dz[depth_i-1]
            depth_AMOC_1000[depth_i]    = depth_AMOC_1000[depth_i-1] + dz_1000[depth_i-1]

        #Depth of water column at the bottom
        H       = np.max(depth_AMOC)

        #Determine the velocity integral
        vel_int     = np.zeros(len(depth_AMOC))

        for depth_i in range(1, len(depth_AMOC)):
            #The velocity profile, v(z=0) = 0, v(z=z_1) = dz * g/(f*rho_0)) * p_b(z=z_1) - p_n(z=z_1), etc.
            vel_int[depth_i] = vel_int[depth_i-1] + dz[depth_i-1]*(self.g/(self.f_cor*self.rho_0))*(PD_south[depth_i-1]-PD_north[depth_i-1])

        #Determine the velocity profile integral, which eventually sets the AMOC
        AMOC_int        = np.zeros(len(depth_AMOC))
        AMOC_int_1000   = np.zeros(len(depth_AMOC))

        for depth_i in range(1, len(depth_AMOC)):
            #The integrated velocities between z=z_n and z=z_n+1
            AMOC_int[depth_i]   = AMOC_int[depth_i-1]   + dz[depth_i-1]*0.5*(vel_int[depth_i] + vel_int[depth_i-1])
            AMOC_int_1000[depth_i]  = AMOC_int_1000[depth_i-1] + dz_1000[depth_i-1]*0.5*(vel_int[depth_i] + vel_int[depth_i-1])

        #Integrations terms
        C   = -AMOC_int[-1]/H 
        D   = 0

        #Vertical profile of overturning (with the correct boundary conditions)
        TWB         = (AMOC_int + C*depth_AMOC + D) / 10**6.0
        
        #Retain the overturning at 1000 m depth
        TWB_1000        = AMOC_int_1000 + C*depth_AMOC_1000 + D
        z_1000_index    = (np.abs(depth_AMOC - 1000)).argmin()
        TWB_1000        = TWB_1000[z_1000_index] / 10**6.0
        
        #Return AMOC strength and the velocity profile
        return depth_AMOC, TWB * self.AMOC_scale, TWB_1000 * self.AMOC_scale
        
    def BOXES(self, field, lat_min, lat_max):
        """Determines the meridionally-averaged values for the given quantity"""
        
        #Get the relevant region
        lat_min_index   = (np.abs(self.lat - lat_min)).argmin()
        lat_max_index   = (np.abs(self.lat - lat_max)).argmin()+1
        field           = field[:, :, lat_min_index:lat_max_index]
        volume_band     = self.lat_volume[:, lat_min_index:lat_max_index]
        
        #Empty array for the average per depth level and the deviations (= tilde) from the vertically-averaged (not volume-averaged!) profiles 
        field_mean  = ma.masked_all((len(field), len(self.depth)))
        field_VA    = ma.masked_all((len(field), len(self.depth)))
        field_tilde = ma.masked_all((len(field), len(self.depth)))
            
        for depth_i in range(len(self.depth)):
            #Loop over each depth level
            volume_depth_norm   = volume_band[depth_i]
            volume_depth_norm   = volume_depth_norm / np.sum(volume_depth_norm)
            field_mean[:, depth_i]  = np.sum(field[:, depth_i] * volume_depth_norm, axis = 1)
            
        #Normalise the layer levels
        dz_norm     = self.dz / np.sum(self.dz)

        for time_i in range(len(field)):
            #Determine the vertically-averaged (not volume-averaged!) profiles and its deviations
            field_VA[time_i]    = np.sum(field_mean[time_i] * dz_norm)  
            field_tilde[time_i] = field_mean[time_i] - field_VA[time_i]
    
        return field_mean, field_VA, field_tilde
        
    def AMOC_SCALING_update(self, AMOC, time_index_1, time_index_2):
        """Updates the AMOC scaling coefficient and the reconstructions, based on the actual AMOC strength"""

        #Calculate the regression scaling coefficient
        model   = sm.OLS(AMOC[time_index_1:time_index_2], self.AMOC_TWB_1000[time_index_1:time_index_2])
        results = model.fit()
        reg     = results.params[0]

        print()
        print('The AMOC scaling coefficient is now rescaled as:', reg)
        print()
        
        self.AMOC_scale = reg
        
        #Now also update the relevant time series as well
        self.AMOC_TWB_z             = self.AMOC_TWB_z * self.AMOC_scale
        self.AMOC_TWB_1000          = self.AMOC_TWB_1000 * self.AMOC_scale
        self.AMOC_TWB_1000_VA       = self.AMOC_TWB_1000_VA * self.AMOC_scale
        self.AMOC_TWB_1000_tilde    = self.AMOC_TWB_1000_tilde * self.AMOC_scale

    def ALPHA_BETA_update(self, TEMP, SALT, RHO):
        """Determines the temperature and salinity contributions for the different boxes"""
        
        #Retain the salinity over the boxes
        self.temp_south, self.temp_south_VA, self.temp_south_tilde  = self.BOXES(TEMP, self.south_lat_1, self.south_lat_2)
        self.temp_north, self.temp_north_VA, self.temp_north_tilde  = self.BOXES(TEMP, self.north_lat_1, self.north_lat_2)
        self.salt_south, self.salt_south_VA, self.salt_south_tilde  = self.BOXES(SALT, self.south_lat_1, self.south_lat_2)
        self.salt_north, self.salt_north_VA, self.salt_north_tilde  = self.BOXES(SALT, self.north_lat_1, self.north_lat_2)
        
        #Determine the derivatives to temperature and salniity
        salt_abs    = ma.masked_all(np.shape(TEMP))
        temp_CT     = ma.masked_all(np.shape(TEMP))
        
        for depth_i in range(len(self.depth)):
            #Get the layer thickness of each layer
            layer_half  = self.dz[depth_i] / 2.0

            if depth_i == 0:
                #First time, determine the pressure over the upper half
                pres    = layer_half * 9.81 * RHO[:, depth_i] / 10**5.0 * 10.

            else:
                #Cummulative add the pressure
                pres    += layer_half * 9.81 * RHO[:, depth_i] / 10**5.0 * 10.  
            
            #Calculate the absolute salinity and conservative temperature
            salt_abs[:, depth_i]    = gsw.SA_from_SP(SALT[:, depth_i], pres, -20, self.lat)
            temp_CT[:, depth_i]     = gsw.CT_from_pt(salt_abs[:, depth_i], TEMP[:, depth_i])

            #Add the remaining half for the pressure 
            pres            += layer_half * 9.81 * RHO[:, depth_i] / 10**5.0 * 10.
            
        #Determine the potential density keeping either the salinity or temperature fixed
        rho_temp    = gsw.sigma0(np.mean(salt_abs[0:50], axis = 0),temp_CT)     #Temperature driven changes
        rho_salt    = gsw.sigma0(salt_abs, np.mean(temp_CT[0:50], axis = 0))    #Salinity driven changes
        
        #Get the same regions for CT temperature and absolute salinity
        temp_CT_south, temp_CT_south_VA, temp_CT_south_tilde        = self.BOXES(temp_CT, self.south_lat_1, self.south_lat_2)
        temp_CT_north, temp_CT_north_VA, temp_CT_north_tilde        = self.BOXES(temp_CT, self.north_lat_1, self.north_lat_2)
        salt_abs_south, salt_abs_south_VA, salt_abs_south_tilde     = self.BOXES(salt_abs, self.south_lat_1, self.south_lat_2)
        salt_abs_north, salt_abs_north_VA, salt_abs_north_tilde     = self.BOXES(salt_abs, self.north_lat_1, self.north_lat_2)

        #Now the density contribution only
        rho_temp_south, rho_temp_south_VA, rho_temp_south_tilde     = self.BOXES(rho_temp, self.south_lat_1, self.south_lat_2)
        rho_temp_north, rho_temp_north_VA, rho_temp_north_tilde     = self.BOXES(rho_temp, self.north_lat_1, self.north_lat_2)
        rho_salt_south, rho_salt_south_VA, rho_salt_south_tilde     = self.BOXES(rho_salt, self.south_lat_1, self.south_lat_2)
        rho_salt_north, rho_salt_north_VA, rho_salt_north_tilde     = self.BOXES(rho_salt, self.north_lat_1, self.north_lat_2)
        
        #Now get the thermal and haline coefficients by fitting the data
        self.alpha  = -linregress(temp_CT_north_VA[:, 0] - temp_CT_south_VA[:, 0], (rho_temp_north_VA[:, 0] - rho_temp_south_VA[:, 0]) / self.rho_0).slope
        self.beta   = linregress(salt_abs_north_VA[:, 0] - salt_abs_south_VA[:, 0], (rho_salt_north_VA[:, 0] - rho_salt_south_VA[:, 0]) / self.rho_0).slope

        print()
        print('Thermal expansion coefficient is now updated as:', self.alpha)
        print('Haline contraction coefficient is now updated as:', self.beta)
        print()
        
        #Now determine the contributions from temperature and salinity alone
        self.AMOC_TWB_1000_VA_temp  = (self.g * self.AMOC_scale) / (self.f_cor) * self.D_scale**2.0 *  -self.alpha * (temp_CT_north_VA[:, 0] - temp_CT_south_VA[:, 0]) / 10**6.0
        self.AMOC_TWB_1000_VA_salt  = (self.g * self.AMOC_scale) / (self.f_cor) * self.D_scale**2.0 *  self.beta * (salt_abs_north_VA[:, 0] - salt_abs_south_VA[:, 0]) / 10**6.0

    def SALT_SCALING_update(self, FovS, window = 10):
        """Determines the salinity scaling coefficient between between the upper and lower AMOC limb"""
    
        #First determine the FovS minimum, which is needed to fit certain parameters
        self.FovS       = FovS
        year_average    = 50
        time_average    = int(len(self.time) / year_average)
        time_average    = ma.masked_all((year_average, time_average))
        FovS_average    = ma.masked_all((year_average, len(time_average[0])))

        for year_start in range(year_average):
            for time_i in range(len(time_average[0])):
                #Loop over each time slice
                if len(self.time[year_start+time_i*year_average:year_start+(time_i+1)*year_average]) != year_average:
                    #No complete period
                    continue

                #Take the mean for different starting years
                time_average[year_start, time_i] = np.mean(self.time[year_start+time_i*year_average:year_start+(time_i+1)*year_average])
                FovS_average[year_start, time_i] = np.mean(FovS[year_start+time_i*year_average:year_start+(time_i+1)*year_average])

        #-----------------------------------------------------------------------------------------  

        FovS_min        = np.zeros(year_average)

        for year_start in range(year_average):
            #Get the different knots
            time_knot       = time_average[year_start]
            FovS_knot       = FovS_average[year_start]
            
            #Get non-masked data
            mask_index      = np.where(time_knot.mask == False)[0]
            time_knot       = time_knot[mask_index]
            FovS_knot       = FovS_knot[mask_index]

            #Fit the cubic splines
            CS          = CubicSpline(time_knot, FovS_knot, bc_type = 'natural')
            FovS_mean   = CS(self.time)
            
            #Get the minimum
            FovS_min[year_start]    = self.time[np.argmin(FovS_mean)]

        #Get the average model year for the FovS minimum
        FovS_min    = np.mean(FovS_min)
        
        print()
        print('FovS minimum at model year:', int(FovS_min))
        print()
        
        #Determine the fit up to the FovS minimum
        self.window     = window
        self.fit_index      = int(FovS_min / self.window)+1
        
        #Determine the upper and lower AMOC limb salinity contrast, including the averages for different variables
        self.salt_up_low= -self.S_0 * self.FovS / self.AMOC_TWB_1000
        self.time_window, self.salt_up_low_window   = self.WindowAverage(self.salt_up_low)
        self.time_window, self.salt_north_VA_window = self.WindowAverage(self.salt_north_VA[:, 0])
        self.time_window, self.salt_south_VA_window = self.WindowAverage(self.salt_south_VA[:, 0])
        self.time_window, self.AMOC_TWB_1000_window = self.WindowAverage(self.AMOC_TWB_1000)
        
        #Determine the scaling coefficient mu
        self.salt_scale, b  = np.polyfit(self.salt_up_low_window[:self.fit_index], self.salt_north_VA_window[:self.fit_index] - self.salt_south_VA_window[:self.fit_index], 1)
        
        #Finally determine the TWB from the salinity contrast
        self.AMOC_TWB_salt_up_low   = (self.salt_scale * self.g * self.AMOC_scale) / (self.f_cor) * self.D_scale**2.0 *  self.beta * self.salt_up_low / 10**6.0 
        
        print('Salinity coefficient relating the meridional salinity difference to the upper and lower AMOC limbs is now updated as:', self.salt_scale)
        print()

    def FeedbackContribution(self, feedback_comp, Vanderborght = False):
        """Determines the strength of the individual feedbacks, requires SALT_SCALING_update"""
        
        if Vanderborght == False:
            #Determine average AMOC and AMOC limb salinity contrast between two windows, use moving windows
            self.time_feedback, AMOC_ma, AMOC_der               = self.FeedbackSmoothing(self.AMOC_TWB_1000)
            self.time_feedback, salt_up_low_ma, salt_up_low_der = self.FeedbackSmoothing(self.salt_up_low)
            self.time_feedback, feedback_ma, feedback_der       = self.FeedbackSmoothing(feedback_comp)
        
            #Calculate the feedback coefficient
            feedback_coef   = self.salt_scale * self.g * self.S_0 * self.AMOC_scale * self.beta * self.D_scale**2.0 / (AMOC_ma * self.f_cor)

            #Determine the window average for the given contribution
            feedback_comp   = feedback_coef * np.cumsum(feedback_der) / 10**6.0
            
            #Always calculate the salt-advection feedback
            self.salt_advection_feedback    = feedback_coef * np.cumsum(-AMOC_der * salt_up_low_ma / self.S_0)/ 10**6.0
            
            return feedback_comp
        
        else:
            #Similar procedure as above, but is discretised using 10-year windows (to check for exact match)"""
        
            #Determine average AMOC and AMOC limb salinity contrast between two windows
            self.time_feedback      = 0.5 * (self.time_window[1:]+ self.time_window[:-1])
            AMOC_window_mean        = 0.5 * (self.AMOC_TWB_1000_window[1:]+ self.AMOC_TWB_1000_window[:-1])
            salt_up_low_window_mean = 0.5 * (self.salt_up_low_window[1:]+ self.salt_up_low_window[:-1])

            #Calculate the feedback coefficient
            feedback_coef           = self.salt_scale * self.g * self.S_0 * self.AMOC_scale * self.beta * self.D_scale**2.0 / (AMOC_window_mean * self.f_cor)
            
            #Determine the window average for the given contribution
            self.time_window, feedback_comp_window      = self.WindowAverage(feedback_comp)
            feedback_comp           = feedback_coef * np.cumsum(np.diff(feedback_comp_window)) / 10**6.0
            
            #Always calculate the salt-advection feedback
            self.salt_advection_feedback            = feedback_coef * np.cumsum(-np.diff(self.AMOC_TWB_1000_window) * salt_up_low_window_mean / self.S_0)/ 10**6.0
            
            return feedback_comp
    
    def ReducedModel(self, AMOC, FovS, FW_gyre, FW_eddy, FovN):
        """Estimate the saddle-node bifurcation using the reduced model from van Westen et al. (2025):
        A saddle-node bifurcation may be causing the AMOC collapse in the Community Earth System Model"""
        
        #Get the scale coefficient between temperature and salinity responses
        self.time_window, self.AMOC_TWB_1000_VA_temp_window = self.WindowAverage(self.AMOC_TWB_1000_VA_temp - self.AMOC_TWB_1000_VA_temp[0])
        self.time_window, self.AMOC_TWB_1000_VA_salt_window = self.WindowAverage(self.AMOC_TWB_1000_VA_salt - self.AMOC_TWB_1000_VA_salt[0])
        
        c_1, b  = np.polyfit(self.AMOC_TWB_1000_VA_salt_window[:self.fit_index], self.AMOC_TWB_1000_VA_temp_window[:self.fit_index], 1)
        c_2 = self.salt_scale * self.g * self.AMOC_scale * self.D_scale**2.0 *  self.beta / self.f_cor / 10**6.0
        
        #Relation between gyre and the vertical salinity contrast
        self.time_window, FW_gyre_window    = self.WindowAverage(FW_gyre)
        g_1, g_2= np.polyfit(self.salt_up_low_window[:self.fit_index], FW_gyre_window[:self.fit_index], 1)
        g_1 = -g_1

        #Relation between eddies and the vertical salinity contrast
        self.time_window, FW_eddy_window    = self.WindowAverage(FW_eddy)
        e_1, e_2= np.polyfit(self.salt_up_low_window[:self.fit_index], FW_eddy_window[:self.fit_index], 1)
        e_1 = -e_1

        #Relation between FovN and the vertical salinity contrast
        self.time_window, FovN_window       = self.WindowAverage(FovN)
        n_1, n_2= np.polyfit(self.salt_up_low_window[:self.fit_index], FovN_window[:self.fit_index], 1)
            
        print()
        print('Fitting terms with the vertical salinity difference')
        print('Gyres :', g_1)
        print('Eddies:', e_1)
        print('FovN  :', n_1)
        print() 
        
        #Get the initial AMOC strength and FovS (first 50 model years or adjust)
        AMOC_init   = np.mean(self.AMOC_TWB_1000[0:50])
        FovS_init   = np.mean(FovS[0:50])
        
        #Constant of salinity and temperaure AMOC responses
        con_salt_temp   = -c_2 * (1 + c_1) * self.S_0   
        
        #Also fit the time series to compare to the constants
        self.time_window, self.AMOC_TWB_1000_VA_window  = self.WindowAverage(self.AMOC_TWB_1000_VA)
        nu_fit, b = np.polyfit(self.salt_up_low_window[:self.fit_index], self.AMOC_TWB_1000_VA_window[:self.fit_index], 1)
        
        print('Constant between AMOC and vertical salinity difference, from the constants:', c_2 * (1 + c_1))
        print('Constant between AMOC and vertical salinity difference, from the fit:', nu_fit)  
        print() 
        
        #Uncomment the line below to get the AMOC tipping estimate with the fitted data
        #con_salt_temp  = -nu_fit * 35.0

        #The base part for both solutions
        AMOC_base   = (AMOC_init / 2.0) - (con_salt_temp * FovS_init / (2 * AMOC_init)) - ((n_1 + g_1 + e_1) * self.S_0 / 2)

        #The critical freshwater flux forcing
        F_H_max     = 1. / con_salt_temp * ((AMOC_init**2.0 + con_salt_temp * FovS_init + (n_1 + g_1 + e_1) * self.S_0 * AMOC_init) / (2 * AMOC_init))**2.0

        #The square root part
        F_H     = np.linspace(0, F_H_max, 100, endpoint = True)
        AMOC_sqrt   = ((AMOC_init**2.0 + con_salt_temp * FovS_init + (n_1 + g_1 + e_1) * self.S_0 * AMOC_init) / (2 * AMOC_init))**2.0 - con_salt_temp * F_H
        
        AMOC_on     = AMOC_base + np.sqrt(AMOC_sqrt)
        AMOC_un     = AMOC_base - np.sqrt(AMOC_sqrt)
        
        #Remove the masked elements
        AMOC_on     = ma.masked_where(AMOC_on != AMOC_on, AMOC_on)
        AMOC_un     = ma.masked_where(AMOC_un != AMOC_un, AMOC_un)
        
        #Note that the factor 0.86 is specific to the CESM hosing set up
        #The hosing strength is linearly related to model year, the factor 0.0003
        #!!!UPDATE WHEN USING A DIFFERENT HOSING PROTOCOL!!!
        print('Critical freshwater flux forcing:', F_H_max * (1 / 0.86))
        
        fig, ax = subplots()
        
        ax.plot(self.time * 0.0003, AMOC, '-k', linewidth = 0.5, alpha = 0.5)
        ax.plot(F_H * (1 / 0.86), AMOC_on, '-', color = 'firebrick', linewidth = 2.0)
        
        ax.set_xlim(0, np.max(self.time)*0.0003)
        ax.set_ylabel('Volume transport (Sv)')
        ax.set_xlabel('Freshwater flux forcing, $F_H$ (Sv)')
        ax.set_ylim(-2, 22)
        ax.grid()
        
        ax.set_title('AMOC strength and estimate')

            
    def WindowAverage(self, data_year):
        """Takes window averages of yearly-averaged time series"""
        
        time_2      = np.zeros(int(len(self.time) / self.window))
        data_2      = ma.masked_all((len(time_2)))

        for time_i in range(len(time_2)):
            time_2[time_i]      = np.mean(self.time[time_i*self.window:(time_i+1)*self.window])
            data_2[time_i]      = np.mean(data_year[time_i*self.window:(time_i+1)*self.window])
            
        return time_2, data_2

    def FeedbackSmoothing(self, data_year, moving_average = 25):
        """Determine the AMOC feedback, smoothed versions,
        determine the mean and derivatives"""
        
        time_2      = np.zeros(len(self.time) - moving_average + 1)
        data_2      = ma.masked_all((len(time_2)))
        data_der_2  = ma.masked_all((len(time_2)))
        
        for time_i in range(len(time_2)):
            time_2[time_i]      = self.time[time_i] + (moving_average-1) / 2.0
            data_2[time_i]      = np.mean(data_year[time_i:time_i+moving_average], axis = 0)
            data_der_2[time_i], b   = np.polyfit(np.arange(moving_average), data_year[time_i:time_i+moving_average], 1)

        return time_2, data_2, data_der_2
        
        
    def PlotVerticalProfiles(self):
        """Plots the meridional differences for potential density, salinity and temperature"""
        
        #Crop the depth
        depth_crop					                  = 1000
        factor_depth_crop			                  = 4
        depth_plot					                  = np.copy(self.depth)
        depth_TWB_plot				                  = np.copy(self.depth_TWB) 
        depth_plot[depth_plot > depth_crop] 		  = ((depth_plot[depth_plot > depth_crop] - depth_crop) / factor_depth_crop) + depth_crop
        depth_TWB_plot[depth_TWB_plot > depth_crop]  = ((depth_TWB_plot[depth_TWB_plot > depth_crop] - depth_crop) / factor_depth_crop) + depth_crop
        
        
        #-----------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------
        
        fig, ax		= subplots()
        
        graph_1,	= ax.plot(np.mean(self.PD_north[0:50] - self.PD_south[0:50], axis = 0), depth_plot, '-', color = 'k', linewidth = 2, label = r'$\Delta \rho(z)$')
        graph_2,	= ax.plot(np.mean(self.PD_north_VA[0:50] - self.PD_south_VA[0:50], axis = 0), depth_plot, '--', color = 'firebrick', linewidth = 2, label = r'$\Delta \overline{\rho}$')
        graph_3		= ax.fill_betweenx(depth_plot, np.mean(self.PD_north[0:50] - self.PD_south[0:50], axis = 0), np.mean(self.PD_north_VA[0:50] - self.PD_south_VA[0:50], axis = 0), color='royalblue', alpha=0.3)
        
        graph_3 = Patch(facecolor='royalblue', alpha=0.3, label = r'$\Delta \tilde{\rho}(z)$')
                      
        ax.set_xlim(-0.05, 0.3)
        ax.set_xlabel('Meridional potential density difference (kg m$^{-3}$)')
        ax.set_ylim(((5500 - depth_crop) / factor_depth_crop) + depth_crop, -25)
        ax.set_ylabel('Depth (m)')
        ax.grid()
        
        labels_rescaled =  ax.get_yticks()
        for label_i in range(len(labels_rescaled)):
        	if labels_rescaled[label_i] > depth_crop:
        		#Rescale the xlabels
        		labels_rescaled[label_i]	= ((labels_rescaled[label_i] - depth_crop) * factor_depth_crop) + depth_crop
        	
        labels_rescaled	= labels_rescaled.astype(int)
        ax.set_yticklabels(labels_rescaled)
        	
        ax.legend(handles=[graph_1, graph_2, graph_3], loc='lower right', ncol=1, framealpha = 1.0, fontsize = 13)
        
        ax.set_title('b) Meridional potential density difference (model year 1 - 50)')
                
        #-----------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------
        
        fig, ax		= subplots()
        
        graph_1,	= ax.plot(np.mean(self.salt_north[0:50] - self.salt_south[0:50], axis = 0), depth_plot, '-', color = 'k', linewidth = 2, label = r'$\Delta S(z)$')
        graph_2,	= ax.plot(np.mean(self.salt_north_VA[0:50] - self.salt_south_VA[0:50], axis = 0), depth_plot, '--', color = 'firebrick', linewidth = 2, label = r'$\Delta \overline{S}$')
        graph_3		= ax.fill_betweenx(depth_plot, np.mean(self.salt_north[0:50] - self.salt_south[0:50], axis = 0), np.mean(self.salt_north_VA[0:50] - self.salt_south_VA[0:50], axis = 0), color='royalblue', alpha=0.3)
        
        graph_3 = Patch(facecolor='royalblue', alpha=0.3, label = r'$\Delta \tilde{S}(z)$')
                      
        ax.set_xlim(-0.1, 0.8)
        ax.set_xlabel('Meridional salinity difference (g kg$^{-1}$)')
        ax.set_ylim(((5500 - depth_crop) / factor_depth_crop) + depth_crop, -25)
        ax.set_ylabel('Depth (m)')
        ax.grid()
        
        labels_rescaled =  ax.get_yticks()
        for label_i in range(len(labels_rescaled)):
        	if labels_rescaled[label_i] > depth_crop:
        		#Rescale the xlabels
        		labels_rescaled[label_i]	= ((labels_rescaled[label_i] - depth_crop) * factor_depth_crop) + depth_crop
        	
        labels_rescaled	= labels_rescaled.astype(int)
        ax.set_yticklabels(labels_rescaled)
        	
        ax.legend(handles=[graph_1, graph_2, graph_3], loc='lower right', ncol=1, framealpha = 1.0, fontsize = 13)
        
        ax.set_title('d) Meridional salinity difference (model year 1 - 50)')
                
        #-----------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------
        
        fig, ax		= subplots()
        
        graph_1,	= ax.plot(np.mean(self.temp_north[0:50] - self.temp_south[0:50], axis = 0), depth_plot, '-', color = 'k', linewidth = 2, label = r'$\Delta T(z)$')
        graph_2,	= ax.plot(np.mean(self.temp_north_VA[0:50] - self.temp_south_VA[0:50], axis = 0), depth_plot, '--', color = 'firebrick', linewidth = 2, label = r'$\Delta \overline{T}$')
        graph_3		= ax.fill_betweenx(depth_plot, np.mean(self.temp_north[0:50] - self.temp_south[0:50], axis = 0), np.mean(self.temp_north_VA[0:50] - self.temp_south_VA[0:50], axis = 0), color='royalblue', alpha=0.3)
        
        graph_3 = Patch(facecolor='royalblue', alpha=0.3, label = r'$\Delta \tilde{T}(z)$')
                      
        ax.set_xlim(-0.2, 4)
        ax.set_xlabel('Meridional temperature difference ($^{\circ}$C)')
        ax.set_ylim(((5500 - depth_crop) / factor_depth_crop) + depth_crop, -25)
        ax.set_ylabel('Depth (m)')
        ax.grid()
        
        labels_rescaled =  ax.get_yticks()
        for label_i in range(len(labels_rescaled)):
        	if labels_rescaled[label_i] > depth_crop:
        		#Rescale the xlabels
        		labels_rescaled[label_i]	= ((labels_rescaled[label_i] - depth_crop) * factor_depth_crop) + depth_crop
        	
        labels_rescaled	= labels_rescaled.astype(int)
        ax.set_yticklabels(labels_rescaled)
        	
        ax.legend(handles=[graph_1, graph_2, graph_3], loc='lower right', ncol=1, framealpha = 1.0, fontsize = 13)
        
        ax.set_title('f) Meridional temperature difference (model year 1 - 50)')
                
        #-----------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------
         
