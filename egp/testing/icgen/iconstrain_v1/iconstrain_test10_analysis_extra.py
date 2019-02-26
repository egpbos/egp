subbox_halfsize = 10
subbox_offset = np.array([int(target_pos[0]/boxlen*gridsize) - subbox_halfsize, int(target_pos[1]/boxlen*gridsize) - subbox_halfsize, int(target_pos[2]/boxlen*gridsize) - subbox_halfsize])

x_range = slice(subbox_offset[0], subbox_offset[0]+2*subbox_halfsize)
y_range = slice(subbox_offset[1], subbox_offset[1]+2*subbox_halfsize)
z_range = slice(subbox_offset[2], subbox_offset[2]+2*subbox_halfsize)
# subbox = minimize_grid[:,x_range,y_range,z_range]
subbox = minimize_grid[x_range,y_range,z_range]

(np.array(np.unravel_index(subbox.argmin(), subbox.shape))*1.0 + subbox_offset)/gridsize*boxlen

pl.imshow(subbox[:,:,result[2]-subbox_offset[2]-subbox_halfsize].T, extent = (subbox_offset[0]*1.0/gridsize*boxlen, subbox_offset[0]*1.0/gridsize*boxlen+2.0*subbox_halfsize/gridsize*boxlen, subbox_offset[1]*1.0/gridsize*boxlen, subbox_offset[1]*1.0/gridsize*boxlen+2.0*subbox_halfsize/gridsize*boxlen))
pl.imshow(subbox[:,result[1]-subbox_offset[1]-subbox_halfsize].T, extent = (subbox_offset[0]*1.0/gridsize*boxlen, subbox_offset[0]*1.0/gridsize*boxlen+2.0*subbox_halfsize/gridsize*boxlen, subbox_offset[2]*1.0/gridsize*boxlen, subbox_offset[2]*1.0/gridsize*boxlen+2.0*subbox_halfsize/gridsize*boxlen))
