import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree
from netCDF4 import Dataset
from datetime import datetime, timedelta

class Wind():
    def __init__(self):
        raise NotImplementedError
    
    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError
        
    def getu(self,model):
        """
        Return u: a list of two matrices (one for x and one for y).
        u needs to be of the the shape: model.resolution.
        
        Requires the model object, as this can tell the method
        about the grid shape, location and resolution etc, that
        it needs to build u over.
        """
        raise NotImplementedError
    
class WindSimple(Wind):
    def __init__(self,speedx,speedy,speedz=None):
        """
        Same wind direction/speed for all time steps.
        
        speedx = Wind in East direction ("right")
        speedy = Wind in North direction ("up")
        
        This is the direction the wind is going to.
        """
        self.speedx = speedx
        self.speedy = speedy
        self.speedz = speedz

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it [something]x3 array [time,x,y].
        
        Added hack to add 3rd axis to space: If speedz is set in constructor then it returns [time,x,y,z]
        """
        #return np.repeat(np.array([self.speedx,self.speedy])[None,:],len(coords),0)
        if self.speedz is None:
            return np.repeat(np.array([self.speedx,self.speedy])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords[...,1:].shape)
        else:
            return np.repeat(np.array([self.speedx,self.speedy,self.speedz])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords[...,1:].shape)

    def getu(self,model):
        u = []
        u.append(np.full(model.resolution,self.speedx)) #x direction wind
        u.append(np.full(model.resolution,self.speedy)) #y direction wind
        return u
        
class WindSimple1d(Wind):
    def __init__(self,speedx):
        """
        Same wind direction/speed for all time steps.
        
        speedx = Wind speed
        
        This is the direction the wind is going to.
        """
        self.speedx = speedx

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it [something]x2 array [time,x].
        """
        return np.repeat(np.array([self.speedx])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords.shape)
       

    def getu(self,model):
        u = []
        u.append(np.full(model.resolution,self.speedx)) #x direction wind        
        return u        
        
class WindFixU(Wind):
    def __init__(self,u):
        """Class for if you need to set the exact matrices of wind.
        
        u is a list of two matrices (one for x and one for y).
        u needs to be of the the shape: model.resolution."""
        self.u = u

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError        

    def getu(self,model):
        #TODO Add exceptions/checks for shape of u.
        return self.u
        
class WindFromStations():
    def __init__(self,stationdata,time_avg):
        """
        Interpolates between locations and times of stations.
        
        stationdata is of the form:
          time, x, y, wind speed, wind direction
          
        - where x and y and time are in the units used for your model
           - ...
        - wind speed should be in the same units too (e.g. km/hour)
        - wind direction should in degrees (north = 0) [direction wind is going to]
          (angle is positive from N->E, i.e. East is +90)
        
        time_avg - how long each sample in our data averages over (e.g. 1 hour)
        """
        self.stationdata = stationdata
        self.time_avg = time_avg
        
    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError 
        
        
    def getu(self,model):
        ux = []
        uy = []
        for tt in np.linspace(model.boundary[0][0],model.boundary[1][0],model.resolution[0]):
            sliceofstationdata = self.stationdata[(self.stationdata[:,0]>tt-self.time_avg) & (self.stationdata[:,0]<=tt),:]
            xvel = np.cos(np.deg2rad(sliceofstationdata[:,4]))*sliceofstationdata[:,3]
            yvel = np.sin(np.deg2rad(sliceofstationdata[:,4]))*sliceofstationdata[:,3]

            #coords = model.coords.reshape(3,np.prod(model.resolution)).T
            xx=np.linspace(model.boundary[0][1],model.boundary[1][1],model.resolution[1])
            yy=np.linspace(model.boundary[0][2],model.boundary[1][2],model.resolution[2])
            fx = interp2d(sliceofstationdata[:,2],sliceofstationdata[:,1],xvel)
            fy = interp2d(sliceofstationdata[:,2],sliceofstationdata[:,1],yvel)            
            ux.append(fx(xx,yy))
            uy.append(fy(xx,yy))
        return [np.array(ux),np.array(uy)]

class RealWindNearestNeighbour:
    def __init__(self, start_date="2019-10-01", num_days=9):
        """
        RealWindNearestNeighbour
        
        This class loads and averages NASA MERRA-2 wind data over selected days and vertical layers,
        and stores all spatial-temporal wind values in a single Pandas DataFrame. It uses a 
        KD-tree for spatial lookup and finds the temporally closest match using a brute-force 
        search over timestamps.
        
        Use Case:
        - Optimized for quick setup and fast nearest-neighbour lookup.
        - Good for testing and small regions.
        - Easier to implement and interpret.
        
        Limitations:
        - Timestamp matching is approximate (not binned), may be less efficient with very large data.
        - Full dataset stored in memory, which may increase memory usage for long durations or large areas.

        Note:
        Default values for start_date, number of days, vertical layer range,
        and bounding box are set for the 2019–20 bushfire case study, but can
        be easily changed for different experiments or regions. 
        The wind data would also need to be downloaded or can be accessed through Earthdata API. 
        https://disc.gsfc.nasa.gov/datasets/M2T3NVASM_5.12.4/summary
        """

        self.proj = Proj(proj='utm', zone=56, south=True, ellps='WGS84')
        self.wind_table = []  # List to accumulate wind data

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        layer_range = (56, 68)  # Height range for averaging wind speeds
        bounding_box = (140.5, -39, 150, -34)

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y%m%d')
            file_path = rf"C:\\Users\\Nur Izfarwiza\\Documents\\Dissertation\\Wind\\MERRA2_400.tavg3_3d_asm_Nv.{date_str}.nc4"

            try:
                dataset = Dataset(file_path, 'r')
                lats = dataset.variables['lat'][:]
                lons = dataset.variables['lon'][:]
                
                # Apply the bounding box filter
                lat_indices = np.where((lats >= bounding_box[1]) & (lats <= bounding_box[3]))[0]
                lon_indices = np.where((lons >= bounding_box[0]) & (lons <= bounding_box[2]))[0]

                lats = lats[lat_indices]
                lons = lons[lon_indices]

                # Load and filter wind data within the bounding box
                eastward_wind = dataset.variables['U'][:, :, lat_indices, :][:, :, :, lon_indices]
                northward_wind = dataset.variables['V'][:, :, lat_indices, :][:, :, :, lon_indices]

                # Average over the specified vertical layers
                layer_range_slice = slice(layer_range[0], layer_range[1])
                eastward_wind_avg = np.mean(eastward_wind[:, layer_range_slice, :, :], axis=1)
                northward_wind_avg = np.mean(northward_wind[:, layer_range_slice, :, :], axis=1)

                # Read and store time directly in minutes
                print("NASA Time Units:", dataset.variables['time'].units)
                time_var = np.array(dataset.variables['time'][:], dtype=np.float64)  # Already in minutes!

                for t_idx, t_val in enumerate(time_var):
                    if not np.isfinite(t_val):
                        continue  # Skip invalid time values
                    timestamp = t_val *60 #convert minutes to seconds

                    # Vectorized coordinate transformation
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    easting, northing = self.proj(lon_grid, lat_grid)

                    # Flatten arrays to create a table of points
                    easting = easting.flatten()
                    northing = northing.flatten()
                    u_wind = eastward_wind_avg[t_idx].flatten()
                    v_wind = northward_wind_avg[t_idx].flatten()

                    # Filter out invalid projections
                    valid_mask = np.isfinite(easting) & np.isfinite(northing) & np.isfinite(u_wind) & np.isfinite(v_wind)

                    # Append valid data to the wind table
                    self.wind_table.extend(
                        zip([timestamp] * np.sum(valid_mask),
                            northing[valid_mask],
                            easting[valid_mask],
                            u_wind[valid_mask],
                            v_wind[valid_mask])
                    )

                dataset.close()

            except FileNotFoundError:
                print(f"File for {date_str} not found.")
            except Exception as e:
                print(f"Error loading {date_str}: {e}")

        # Convert to pandas DataFrame for fast lookups
        self.wind_table = pd.DataFrame(self.wind_table, columns=["Timestamp", "Northing", "Easting", "East Wind", "North Wind"])

        # Clean data: Drop NaN or Inf values
        self.wind_table = self.wind_table.replace([np.inf, -np.inf], np.nan)
        self.wind_table = self.wind_table.dropna()

        # Sort by timestamp for faster temporal queries
        self.wind_table = self.wind_table.sort_values(by="Timestamp").reset_index(drop=True)

        # Build a spatial KD-Tree for efficient spatial querying
        self.wind_tree = cKDTree(self.wind_table[["Easting", "Northing"]].values)

        print(f"Wind data precomputed for {len(self.wind_table)} points across {num_days} days.")
    
    def getwind(self, coords):
        """
        Get the nearest-neighbor wind speed for particles at given positions and times.

        Parameters:
        - coords: A tensor of shape (num_particles, num_observations, 3)
                  where each entry is [time, easting, northing].

        Returns:
        - wind_data: A tensor of shape (num_particles, num_observations, 2)
                     where each entry is [east_wind_speed, north_wind_speed].
        """
        num_particles, num_observations, _ = coords.shape
        wind_data = np.full((num_particles, num_observations, 2), np.nan)  # Initialize with NaN

        timestamps = coords[:, :, 0].flatten()  # Keep time in minutes

        for i in range(num_particles):
            for j in range(num_observations):
                easting, northing = coords[i, j, 1], coords[i, j, 2]
                timestamp = timestamps[i * num_observations + j]

                closest_time_idx = (np.abs(self.wind_table["Timestamp"] - timestamp)).idxmin()
                dist, closest_idx = self.wind_tree.query([easting, northing], k=1)

                wind_data[i, j] = self.wind_table.loc[closest_idx, ["East Wind", "North Wind"]]

        return wind_data


class RealWindBinned:
    """
    RealWindBinned
    
    This class provides an optimized wind lookup method using pre-binned timestamps and per-time-step
    KD-trees for spatial lookup. It loads NASA MERRA-2 wind data, averages over specified vertical layers,
    and groups wind fields into 3-hour intervals for fast matching.
    
    Use Case:
    - More efficient for large datasets and many particles over long time periods.
    - Reduces temporal search time by using fixed 3-hour bins (MERRA-2 resolution).
    - Better spatial query performance with separate KD-trees per timestamp.
    
    Limitations:
    - Slightly more complex implementation.
    - Requires timestamp rounding to nearest 3-hour interval to match wind records.
    """
    def __init__(self, start_date="2019-10-01", num_days=9):
        self.proj = Proj(proj='utm', zone=56, south=True, ellps='WGS84')
        self.wind_by_time = {}
        self.kdtrees = {}

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        layer_range = (56, 68)
        bounding_box = (110, -45, 155, -10)

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y%m%d')
            file_path = rf"C:\\Users\\Nur Izfarwiza\\Documents\\Dissertation\\Wind\\MERRA2_400.tavg3_3d_asm_Nv.{date_str}.nc4"

            try:
                dataset = Dataset(file_path, 'r')
                lats = dataset.variables['lat'][:]
                lons = dataset.variables['lon'][:]

                lat_indices = np.where((lats >= bounding_box[1]) & (lats <= bounding_box[3]))[0]
                lon_indices = np.where((lons >= bounding_box[0]) & (lons <= bounding_box[2]))[0]

                lats = lats[lat_indices]
                lons = lons[lon_indices]

                eastward_wind = dataset.variables['U'][:, :, lat_indices, :][:, :, :, lon_indices]
                northward_wind = dataset.variables['V'][:, :, lat_indices, :][:, :, :, lon_indices]

                layer_range_slice = slice(layer_range[0], layer_range[1])
                eastward_wind_avg = np.mean(eastward_wind[:, layer_range_slice, :, :], axis=1)
                northward_wind_avg = np.mean(northward_wind[:, layer_range_slice, :, :], axis=1)

                time_var = np.array(dataset.variables['time'][:], dtype=np.float64)

                for t_idx, t_val in enumerate(time_var):
                    if not np.isfinite(t_val):
                        continue
                    timestamp = int((t_val * 60) // 10800 * 10800)  # Round to nearest 3-hour (in seconds)

                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    easting, northing = self.proj(lon_grid, lat_grid)

                    easting = easting.flatten()
                    northing = northing.flatten()
                    u_wind = eastward_wind_avg[t_idx].flatten()
                    v_wind = northward_wind_avg[t_idx].flatten()

                    valid_mask = np.isfinite(easting) & np.isfinite(northing) & np.isfinite(u_wind) & np.isfinite(v_wind)
                    data = np.stack([easting[valid_mask], northing[valid_mask], u_wind[valid_mask], v_wind[valid_mask]], axis=1)

                    if timestamp not in self.wind_by_time:
                        self.wind_by_time[timestamp] = data
                    else:
                        self.wind_by_time[timestamp] = np.vstack([self.wind_by_time[timestamp], data])

                dataset.close()

            except Exception as e:
                print(f"Error loading {date_str}: {e}")

        for ts_key, data in self.wind_by_time.items():
            self.kdtrees[ts_key] = cKDTree(data[:, :2])

        print(f"Loaded wind data for {len(self.wind_by_time)} timestamps.")

    def getwind(self, coords):
        num_particles, num_observations, _ = coords.shape
        wind_data = np.full((num_particles, num_observations, 2), np.nan)

        flat_coords = coords.reshape(-1, 3)
        coords_flat = coords.reshape(-1, 3)
        times = coords_flat[:, 0]
        eastings = coords_flat[:, 1]
        northings = coords_flat[:, 2]
    
        # Define bounding box in UTM (convert once at init for faster runtime)
        # Example: UTM for (110E, -45) to (155E, -10)
        x_min, y_min = self.proj(110, -45)
        x_max, y_max = self.proj(155, -10)
    
        for idx, (t, x, y) in enumerate(zip(times, eastings, northings)):
            if not np.isfinite(x) or not np.isfinite(y):
                print(f"Particle {idx} has NaN or Inf: x={x}, y={y}")
                continue
    
            if x < x_min or x > x_max or y < y_min or y > y_max:
                print(f"⚠️ Particle {idx} out of bounds:")
                print(f"  Time: {t:.2f} s | Easting: {x:.2f} | Northing: {y:.2f}")
                continue
        timestamps = (flat_coords[:, 0] // 10800 * 10800).astype(int)
        unique_ts = np.unique(timestamps)

        for ts in unique_ts:
            if ts not in self.kdtrees:
                continue

            mask = timestamps == ts
            coords_subset = flat_coords[mask]
            spatial_coords = coords_subset[:, 1:3]

            tree = self.kdtrees[ts]
            data = self.wind_by_time[ts]

            _, idx = tree.query(spatial_coords)
            wind_vals = data[idx, 2:]

            wind_data.reshape(-1, 2)[mask] = wind_vals

        return wind_data


class RealWindHybrid:
    def __init__(self, start_date="2019-10-01", num_days=1):
        self.proj = Proj(proj='utm', zone=56, south=True, ellps='WGS84')
        self.wind_by_time = {}
        self.kdtrees = {}

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        layer_range = (56, 68)
        bounding_box = (110, -45, 155, -10)

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y%m%d')
            file_path = rf"C:\Users\Nur Izfarwiza\Documents\Dissertation\Wind\MERRA2_400.tavg3_3d_asm_Nv.{date_str}.nc4"

            try:
                dataset = Dataset(file_path, 'r')
                lats = dataset.variables['lat'][:]
                lons = dataset.variables['lon'][:]

                lat_indices = np.where((lats >= bounding_box[1]) & (lats <= bounding_box[3]))[0]
                lon_indices = np.where((lons >= bounding_box[0]) & (lons <= bounding_box[2]))[0]

                lats = lats[lat_indices]
                lons = lons[lon_indices]

                eastward_wind = dataset.variables['U'][:, :, lat_indices, :][:, :, :, lon_indices]
                northward_wind = dataset.variables['V'][:, :, lat_indices, :][:, :, :, lon_indices]

                eastward_wind_avg = np.mean(eastward_wind[:, layer_range[0]:layer_range[1], :, :], axis=1)
                northward_wind_avg = np.mean(northward_wind[:, layer_range[0]:layer_range[1], :, :], axis=1)

                time_var = np.array(dataset.variables['time'][:], dtype=np.float64)

                for t_idx, t_val in enumerate(time_var):
                    timestamp = int(t_val * 60)
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    easting, northing = self.proj(lon_grid, lat_grid)

                    u_wind = eastward_wind_avg[t_idx].flatten()
                    v_wind = northward_wind_avg[t_idx].flatten()
                    easting = easting.flatten()
                    northing = northing.flatten()

                    valid_mask = np.isfinite(easting) & np.isfinite(northing) & np.isfinite(u_wind) & np.isfinite(v_wind)
                    data = np.stack([easting[valid_mask], northing[valid_mask], u_wind[valid_mask], v_wind[valid_mask]], axis=1)

                    self.wind_by_time[timestamp] = data

                dataset.close()

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        for ts, data in self.wind_by_time.items():
            self.kdtrees[ts] = cKDTree(data[:, :2])

        self.available_times = sorted(self.wind_by_time.keys())

    def interpolate_wind(self, t, x, y):
        times = self.available_times
        t_lower = max([ts for ts in times if ts <= t], default=None)
        t_upper = min([ts for ts in times if ts >= t], default=None)

        if t_lower is not None and t_upper is not None and t_lower != t_upper:
            tree_lower = self.kdtrees[t_lower]
            data_lower = self.wind_by_time[t_lower]
            _, idx_lower = tree_lower.query([x, y])
            wind_lower = data_lower[idx_lower, 2:]

            tree_upper = self.kdtrees[t_upper]
            data_upper = self.wind_by_time[t_upper]
            _, idx_upper = tree_upper.query([x, y])
            wind_upper = data_upper[idx_upper, 2:]

            alpha = (t - t_lower) / (t_upper - t_lower)
            return (1 - alpha) * wind_lower + alpha * wind_upper
        elif t_lower is not None:
            tree = self.kdtrees[t_lower]
            data = self.wind_by_time[t_lower]
            _, idx = tree.query([x, y])
            return data[idx, 2:]
        elif t_upper is not None:
            tree = self.kdtrees[t_upper]
            data = self.wind_by_time[t_upper]
            _, idx = tree.query([x, y])
            return data[idx, 2:]
        else:
            return np.array([0.0, 0.0])  # Fallback

    def getwind(self, coords):
        num_particles, num_obs, _ = coords.shape
        wind_output = np.full((num_particles, num_obs, 2), np.nan)

        for i in range(num_particles):
            for j in range(num_obs):
                t, x, y = coords[i, j]
                wind_output[i, j] = self.interpolate_wind(t, x, y)

        return wind_output


class FastWindGrid:
    def __init__(self, start_date="2019-10-01", num_days=1):
        self.proj = Proj(proj='utm', zone=56, south=True, ellps='WGS84')
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.layer_range = (56, 68)
        self.bounding_box = (140.5, -39, 150, -34)

        # For now we assume constant grid
        self.time_vals = []
        self.x_vals = None
        self.y_vals = None
        self.wind_u = []
        self.wind_v = []

        for day in range(num_days):
            date = self.start_date + timedelta(days=day)
            date_str = date.strftime('%Y%m%d')
            filepath = rf"C:\Users\Nur Izfarwiza\Documents\Dissertation\Wind\MERRA2_400.tavg3_3d_asm_Nv.{date_str}.nc4"
            ds = Dataset(filepath)

            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:]
            lat_idx = np.where((lats >= self.bounding_box[1]) & (lats <= self.bounding_box[3]))[0]
            lon_idx = np.where((lons >= self.bounding_box[0]) & (lons <= self.bounding_box[2]))[0]
            lats = lats[lat_idx]
            lons = lons[lon_idx]

            # Convert to UTM
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            x_grid, y_grid = self.proj(lon_grid, lat_grid)

            if self.x_vals is None:
                self.x_vals = x_grid[0, :]
                self.y_vals = y_grid[:, 0]

            U = ds.variables['U'][:, self.layer_range[0]:self.layer_range[1], lat_idx, :][:, :, :, lon_idx]
            V = ds.variables['V'][:, self.layer_range[0]:self.layer_range[1], lat_idx, :][:, :, :, lon_idx]
            u_mean = np.mean(U, axis=1)
            v_mean = np.mean(V, axis=1)

            times = ds.variables['time'][:]
            for t_idx, t in enumerate(times):
                timestamp = int((t * 60))  # seconds
                self.time_vals.append(timestamp)
                self.wind_u.append(u_mean[t_idx])
                self.wind_v.append(v_mean[t_idx])

            ds.close()

        # Convert to 3D arrays: [time, y, x]
        self.wind_u = np.stack(self.wind_u)  # shape (T, Y, X)
        self.wind_v = np.stack(self.wind_v)
        self.time_vals = np.array(self.time_vals)

        print(f"✅ Loaded {len(self.time_vals)} timestamps with shape {self.wind_u.shape}")


    def getwind(self, coords):
        """
        coords: shape (N_particles, N_obs, 3) in [time, easting, northing]
        Returns: (N_particles, N_obs, 2)
        """
        coords_flat = coords.reshape(-1, 3)
        times = coords_flat[:, 0]
        x = coords_flat[:, 1]
        y = coords_flat[:, 2]

        # Convert to grid index
        t_idx = np.argmin(np.abs(self.time_vals[:, None] - times[None, :]), axis=0)

        x_idx = np.floor((x - self.x_vals[0]) / (self.x_vals[1] - self.x_vals[0])).astype(int)
        y_idx = np.floor((y - self.y_vals[0]) / (self.y_vals[1] - self.y_vals[0])).astype(int)

        # Clip indices to valid range
        x_idx = np.clip(x_idx, 0, len(self.x_vals) - 1)
        y_idx = np.clip(y_idx, 0, len(self.y_vals) - 1)
        t_idx = np.clip(t_idx, 0, len(self.time_vals) - 1)

        # Lookup
        u = self.wind_u[t_idx, y_idx, x_idx]
        v = self.wind_v[t_idx, y_idx, x_idx]

        wind = np.stack([u, v], axis=-1).reshape(coords.shape[0], coords.shape[1], 2)
        return wind




