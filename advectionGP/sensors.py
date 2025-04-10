import numpy as np

class SensorModel():
    def __init__(self):
        """Builds H"""
        assert False, "Not implemented" #TODO Turn into an exception
        
    def getHs(self):
        assert False, "Not implemented" #TODO Turn into an exception
        
    #Can also implement:
    #genParticlesFromObservations(self,Nparticles)
    
class FixedSensorModel(SensorModel):
    def __init__(self,obsLocations,spatialAveraging):
        """Return a self.resolution array describing how the concentration is added up for an observation in x.
        Uses self.spatial_averaging to extend the part of the domain that is being observed.
        
        Parameters:
            x == a 4 element vector, time_start, time_end, x, y
            
        The getHs method returns a model.resolution sized numpy array
        """
        self.obsLocs = obsLocations
        self.spatialAveraging = spatialAveraging
        #TO DO
       
    def getHs(self,model):
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)            
        """

        halfGridTile = np.full(model.N_D,self.spatialAveraging/2)
        halfGridTile[0] = 0
        #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startObs = np.delete(self.obsLocs,1,1)
        endObs = np.delete(self.obsLocs,0,1)
        startOfHs = model.getGridCoord(startObs-halfGridTile)
        endOfHs = model.getGridCoord(endObs+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        #print(startOfHs,endOfHs)
        assert (np.all(startObs-halfGridTile>=model.boundary[0])) & (np.all(startObs-halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(endObs+halfGridTile>=model.boundary[0])) & (np.all(endObs+halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
        
        delta,_ = model.getGridStepSize()

        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            
            if len(start)==3:
                h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/((end[0]-start[0])*(end[1]-start[1])*(end[2]-start[2])*np.prod(delta))                
            if len(start)==2:
                h[start[0]:end[0],start[1]:end[1]] = 1/((end[0]-start[0])*(end[1]-start[1])*np.prod(delta))
            if len(start)==1:
                h[start[0]:end[0]] = 1/((end[0]-start[0])*np.prod(delta))
                
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/((end[0]-start[0])*(end[1]-start[1])*(end[2]-start[2])*(dt*dx*dy))
            #h /= (np.sum(h)*dt*dx*dy)
            #print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
            
    def getHs2D(self,model):
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)
            
        """
        halfGridTile = np.array([0,self.spatialAveraging/2])
        #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startOfHs = model.getGridCoord(self.obsLocs[:,[0,2]]-halfGridTile)
        endOfHs = model.getGridCoord(self.obsLocs[:,[1,2]]+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        #print(startOfHs,endOfHs)
        assert (np.all(self.obsLocs[:,[0,2]]-halfGridTile>=model.boundary[0])) & (np.all(self.obsLocs[:,[0,2]]-halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(self.obsLocs[:,[1,2]]+halfGridTile>=model.boundary[0])) & (np.all(self.obsLocs[:,[1,2]]+halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
                
        dt,dx,dx2,Nt,Nx = model.getGridStepSize()
        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            h[start[0]:end[0],start[1]:end[1]] = 1/((end[0]-start[0])*(end[1]-start[1])*(dt*dx))
            #h /= (np.sum(h)*dt*dx*dy)
            #print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
            
    def getHs1D(self,model):
            """
            Returns an iterator providing indicator matrices for each observation on a 1D grid.
            Should integrate to one over the `actual' space (but not necessarily over the grid).
            Arguments:
                model == is a model object (provides grid resolution etc)

            """
            
            #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
            startOfHs = model.getGridCoord(self.obsLocs[:,[0]]) # start of observation ranges
            endOfHs = model.getGridCoord(self.obsLocs[:,[1]]) # end of observartion ranges
            
            endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.

            assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
            assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
            assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
            dt,dt2,Nt = model.getGridStepSize()
            for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
                h = np.zeros(model.resolution)
                #h[start[0]:end[0]] = 1/(tlength) 
                h[start[0]:end[0]] = 1/((end[0]-start[0])*dt) 
                yield h
        
class FixedSensorModel(SensorModel):
    def __init__(self,obsLocations,spatialAveraging):
        """Return a self.resolution array describing how the concentration is added up for an observation in x.
        Uses self.spatial_averaging to extend the part of the domain that is being observed.
        
        Parameters:
            x == a 4 element vector, time_start, time_end, x, y
            
        The getHs method returns a model.resolution sized numpy array
        """
        self.obsLocs = obsLocations
        self.spatialAveraging = spatialAveraging
        #TO DO
       
    def getHs(self,model):
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)            
        """

        halfGridTile = np.full(model.N_D,self.spatialAveraging/2)
        halfGridTile[0] = 0
        #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startObs = np.delete(self.obsLocs,1,1)
        endObs = np.delete(self.obsLocs,0,1)
        startOfHs = model.getGridCoord(startObs-halfGridTile)
        endOfHs = model.getGridCoord(endObs+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        #print(startOfHs,endOfHs)
        assert (np.all(startObs-halfGridTile>=model.boundary[0])) & (np.all(startObs-halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(endObs+halfGridTile>=model.boundary[0])) & (np.all(endObs+halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
        
        delta,_ = model.getGridStepSize()

        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            
            if len(start)==3:
                h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/((end[0]-start[0])*(end[1]-start[1])*(end[2]-start[2])*np.prod(delta))                
            if len(start)==2:
                h[start[0]:end[0],start[1]:end[1]] = 1/((end[0]-start[0])*(end[1]-start[1])*np.prod(delta))
            if len(start)==1:
                h[start[0]:end[0]] = 1/((end[0]-start[0])*np.prod(delta))
                
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/((end[0]-start[0])*(end[1]-start[1])*(end[2]-start[2])*(dt*dx*dy))
            #h /= (np.sum(h)*dt*dx*dy)
            #print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
            
    def getHs2D(self,model):
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)
            
        """
        halfGridTile = np.array([0,self.spatialAveraging/2])
        #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startOfHs = model.getGridCoord(self.obsLocs[:,[0,2]]-halfGridTile)
        endOfHs = model.getGridCoord(self.obsLocs[:,[1,2]]+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        #print(startOfHs,endOfHs)
        assert (np.all(self.obsLocs[:,[0,2]]-halfGridTile>=model.boundary[0])) & (np.all(self.obsLocs[:,[0,2]]-halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(self.obsLocs[:,[1,2]]+halfGridTile>=model.boundary[0])) & (np.all(self.obsLocs[:,[1,2]]+halfGridTile<=model.boundary[1])), "Observation cell isn't inside the grid."
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
                
        dt,dx,dx2,Nt,Nx = model.getGridStepSize()
        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            #h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            h[start[0]:end[0],start[1]:end[1]] = 1/((end[0]-start[0])*(end[1]-start[1])*(dt*dx))
            #h /= (np.sum(h)*dt*dx*dy)
            #print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
            
    def getHs1D(self,model):
            """
            Returns an iterator providing indicator matrices for each observation on a 1D grid.
            Should integrate to one over the `actual' space (but not necessarily over the grid).
            Arguments:
                model == is a model object (provides grid resolution etc)

            """
            
            #print(self.obsLocs[:,[0,2,3]]-halfGridTile)
            startOfHs = model.getGridCoord(self.obsLocs[:,[0]]) # start of observation ranges
            endOfHs = model.getGridCoord(self.obsLocs[:,[1]]) # end of observartion ranges
            
            endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.

            assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
            assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
            assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
            dt,dt2,Nt = model.getGridStepSize()
            for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
                h = np.zeros(model.resolution)
                #h[start[0]:end[0]] = 1/(tlength) 
                h[start[0]:end[0]] = 1/((end[0]-start[0])*dt) 
                yield h
        
class RemoteSensingModel(SensorModel):
    def __init__(self, num_particles=10, spatial_averaging=50000):
        self.num_particles = num_particles
        self.spatialAveraging = spatial_averaging

        # Fixed bounding box over Victoria (lon/lat)
        self.bounding_box = (140.5, -39, 150, -34)
        self.proj = Proj(proj='utm', zone=56, south=True, ellps='WGS84')

        # Generate polygons
        self.grid_polygons = self._generate_grid_polygons()

    def _generate_grid_polygons(self):
        file_path = r"C:\Users\Nur Izfarwiza\Documents\Dissertation\Wind\MERRA2_400.tavg3_3d_asm_Nv.20191001.nc4"
        dataset = Dataset(file_path, 'r')
        lats = dataset.variables['lat'][:]
        lons = dataset.variables['lon'][:]
        dataset.close()

        lat_indices = np.where((lats >= self.bounding_box[1]) & (lats <= self.bounding_box[3]))[0]
        lon_indices = np.where((lons >= self.bounding_box[0]) & (lons <= self.bounding_box[2]))[0]
        subset_lats = lats[lat_indices]
        subset_lons = lons[lon_indices]

        lon_grid, lat_grid = np.meshgrid(subset_lons, subset_lats)
        eastings, northings = self.proj(lon_grid, lat_grid)

        grid_size = 1.0
        lat_bins = np.arange(self.bounding_box[1], self.bounding_box[3] + grid_size, grid_size)
        lon_bins = np.arange(self.bounding_box[0], self.bounding_box[2] + grid_size, grid_size)

        polygons = []
        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                coords = np.array([
                    [eastings[i, j],     northings[i, j]],
                    [eastings[i+1, j],   northings[i+1, j]],
                    [eastings[i+1, j+1], northings[i+1, j+1]],
                    [eastings[i, j+1],   northings[i, j+1]],
                    [eastings[i, j],     northings[i, j]]
                ])
                poly = Polygon(coords)
                polygons.append(poly)
        return polygons

    def genParticles(self, Nparticles=None, t_start=1260, t_end=1440):
        if Nparticles is None:
            Nparticles = self.num_particles

        all_particles = []

        for poly in self.grid_polygons:
            spatial_pts = pointpats.random.poisson(poly, size=Nparticles)
            times = np.random.uniform(t_start, t_end, size=(Nparticles, 1)) * 60  # seconds
            particle_group = np.hstack([times, spatial_pts])
            all_particles.append(particle_group)

        particles = np.vstack(all_particles)  # shape (Nparticles * Nobs, 3)
        Nobs = len(self.grid_polygons)
        particles = particles.reshape(Nobs, Nparticles, 3).transpose(1, 0, 2)  # shape (Nparticles, Nobs, 3)
        return particles

