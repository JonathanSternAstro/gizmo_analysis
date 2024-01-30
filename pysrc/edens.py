import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import unyt as u

"""
" Cross-product (numpy's version is slower)
"""
def explicitcross(a,b):
    e = np.zeros_like(a)
    if len(a.shape) == 3:
        e[:,:,0] = a[:,:,1]*b[:,:,2] - a[:,:,2]*b[:,:,1]
        e[:,:,1] = a[:,:,2]*b[:,:,0] - a[:,:,0]*b[:,:,2]
        e[:,:,2] = a[:,:,0]*b[:,:,1] - a[:,:,1]*b[:,:,0]
    elif len(a.shape) == 2:
        e[:,0] = a[:,1]*b[:,2] - a[:,2]*b[:,1]
        e[:,1] = a[:,2]*b[:,0] - a[:,0]*b[:,2]
        e[:,2] = a[:,0]*b[:,1] - a[:,1]*b[:,0]
    return e

"""
" Find galaxy canter (COM)
"""
def find_center(r,m):
    # Calculate the center of mass
    tot_mass = np.sum(m)
    com_x = np.dot(m, r[:,0])/tot_mass
    com_y = np.dot(m, r[:,1])/tot_mass
    com_z = np.dot(m, r[:,2])/tot_mass
    com = [ com_x, com_y, com_z ]
    return com

"""
" Center the coordinates due to the center of mass
"""
def center_coordinates(r,center):

    r_centered = r - np.tile(center, r.shape[0]).reshape(r.shape)
    return r_centered


"""
" Calculate the angular momentum for an array of particles
"""
def calc_angular_momentum(r,v):
    r = np.array(r)
    v = np.array(v)
    L = explicitcross(r,v)
    L_size = np.sqrt(np.sum(L**2, axis=len(r.shape)-1))
    L_disk = np.sum(L, axis=0)
    return L, L_size, L_disk


"""
" Rotate coordinates such that n_vec will unite with the z axis 
" to rotate to the galaxy disk plane, call with n_vec = L_disk
"""
def rotate_coords(r, n_vec):
    
    # Angles with z and x axes
    R = np.linalg.norm(n_vec)
    theta = np.arccos(n_vec[2]/R)
    phi = np.arctan2(n_vec[1],n_vec[0])

    # Rotation matrices
         # The rounding fixes some small errors in sine and cosine functions 
    c_z =  round(np.cos(2*np.pi-phi),12)  
    s_z = round(np.sin(2*np.pi-phi),12) 
    z_rot_mat = np.array([[c_z,-s_z,0],[s_z,c_z,0],[0,0,1]])

    c_y = round(np.cos(2*np.pi-theta),12)
    s_y = round(np.sin(2*np.pi-theta),12)
    y_rot_mat = np.array([[c_y,0,s_y],[0,1,0],[-s_y,0,c_y]])

    # multiply all particle positions and velocities by rotation matrices
    rotz = np.einsum("ij,kj->ik", r, z_rot_mat)
    roty = np.einsum("ij,kj->ik", rotz, y_rot_mat)
    r_disk_aligned = roty

    return r_disk_aligned


"""
" Convert cartesian to spherical coordinates
"""
def convert_to_spherical(r):
    R = np.sqrt(np.sum(r**2,axis=len(r.shape)-1))
    theta = np.arccos(r[:,2]/R)
    phi = np.arctan2(r[:,1],r[:,0]) 
    theta = theta - 0.5*np.pi # Fix the range to be between -90,90
    return(R,theta,phi)

"""
" Create evenly spaced particles rotating on a disk with normal axis n_vec (for tests). returns positions and velocities arrays
"""

def create_test_disk(n_vec):
    # Create a 2D square of equally spaced dots
    x_test = np.linspace(-10,10,21)
    y_test = np.linspace(-10,10,21)
    X_test,Y_test = np.meshgrid(x_test,y_test)
    
    # Circle it up
    X_test_circ = X_test[X_test**2 + Y_test**2 <= 100]
    Y_test_circ = Y_test[X_test**2 + Y_test**2 <= 100]
    
    # Add the 3rd dimension such that the square is on a plane with normal vector n_vec
    Z_test = - ( n_vec[0] * X_test + n_vec[1] * Y_test ) / n_vec[2]
    Z_test_circ = - ( n_vec[0] * X_test_circ + n_vec[1] * Y_test_circ ) / n_vec[2]
    
    # Combine x y and z
    XYZ_test = np.array([X_test.flatten(),Y_test.flatten(), Z_test.flatten()]).T
    XYZ_circ = np.array([X_test_circ.flatten(),Y_test_circ.flatten(), Z_test_circ.flatten()]).T
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(XYZ_circ[:,0],XYZ_circ[:,1],XYZ_circ[:,2],'.')
    
    # Give the dots circular velocity around the normal axis
    omega = np.tile(n_vec, XYZ_circ.shape[0]).reshape(XYZ_circ.shape)
    V_test = explicitcross(omega,XYZ_circ).reshape(XYZ_circ.shape)
    ax.quiver(XYZ_circ[:,0],XYZ_circ[:,1],XYZ_circ[:,2], V_test[:,0], V_test[:,1], V_test[:,2], color='r', length=0.5, alpha=0.5, arrow_length_ratio=0.0005)

    return( XYZ_circ, V_test )

PARTTYPE= {"gas": "PartType0", "stars": "PartType4"}

KEYS_FIRE = ['Coordinates', 'Density', 'ElectronAbundance', 'InternalEnergy', 'Masses', 'Metallicity', 'NeutralHydrogenAbundance', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber', 'ParticleIDs', 'Potential', 'SmoothingLength', 'StarFormationRate', 'Velocities']

KEYS_IDEAL =  ['BH_Dist', 'CoolingRate', 'Coordinates', 'Density', 'ElectronAbundance', 'HeatingRate', 'HydroHeatingRate', 'IMFFormationProperties', 'InternalEnergy', 'Masses', 'MetalCoolingRate', 'Metallicity', 'MyHeatT', 'NetHeatingRateQ', 'NeutralHydrogenAbundance', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber', 'ParticleIDs', 'SmoothingLength', 'StarFormationRate', 'Velocities']

# load data from cosmological 
def load_data(snap_path_read, partType=['gas','stars'], keys=['Coordinates','Velocities']):
   
    datadict = dict()

    for part in partType:
        datadict[part] = dict()
        for key in keys:
            datadict[part][key] = []
    
    for snap_filename in os.listdir(snap_path_read):
        if snap_filename.endswith(".hdf5"):    
            snap_path = os.path.join(snap_path_read, snap_filename)
            snap = h5py.File(snap_path, 'r') 

            for type in partType:
                particles = snap[PARTTYPE[type]]
                #print(particles.keys())
                for key in keys:
                    quantity = np.array(particles[key][...]).astype(np.float32)
                    s = quantity.shape[-1]
                    if len(datadict[type][key]) == 0:
                        datadict[type][key] = quantity
                    else:
                        quantity_list = np.append(datadict[type][key], quantity)
                        if quantity.ndim == 2:
                            quantity_list = quantity_list.reshape(-1,s)
                        datadict[type][key] = quantity_list
            snap.close()

    return( datadict )

def temperature(internalEnergy, ElectronAbundance, gamma, helium_mass_fraction):
    y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance) 
    mean_molecular_weight = mu * u.proton_mass
    return(mean_molecular_weight * (gamma-1) * internalEnergy / u.kboltz)

# write back to file (be careful cause it overrides the existing data!!)
def write_to_fire_sim(snap_path_write, r_gas_comb_cen_aligned, v_gas_comb_cen_aligned, r_stars_comb_cen_aligned, v_stars_comb_cen_aligned):
    for snap_filename in os.listdir(snap_path_write):
        if snap_filename.endswith(".hdf5"):
            snap_path = os.path.join(snap_path_write, snap_filename)
            snap = h5py.File(snap_path, 'r+')
            gas = snap['PartType0']
            stars = snap['PartType4']
    
            N_gas = len(gas['Coordinates'])
            N_stars = len(stars['Coordinates'])
            
            gas['Coordinates'][...] = r_gas_comb_cen_aligned[:N_gas]
            gas['Velocities'][...] = v_gas_comb_cen_aligned[:N_gas]
            r_gas_comb_cen_aligned = r_gas_comb_cen_aligned[N_gas:]
            v_gas_comb_cen_aligned = v_gas_comb_cen_aligned[N_gas:]
        
            stars['Coordinates'][...] = r_stars_comb_cen_aligned[:N_stars]
            stars['Velocities'][...] = v_stars_comb_cen_aligned[:N_stars]
            r_stars_comb_cen_aligned = r_stars_comb_cen_aligned[N_stars:]
            v_dat_comb_cen_aligned = v_stars_comb_cen_aligned[N_stars:]
        
        snap.close()