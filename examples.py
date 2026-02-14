#examples

import os
import functions as fun
import descriptors as desc
import matplotlib.pyplot as plt
import numpy as np


# prepare the rhythm space 
##########################

# parse all midi patterns found in a folder (including subfolders)
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments")
print("number of patterns found: ", len(all_pattlists))

# convert themn to the pattlist format and get their names
d = desc.lopl2descriptors(all_pattlists)
print("number of descriptors found: ", len(d))

# create positions from descriptors using embedding
pos = fun.d_mtx_2_mds(d)
print("number of positions found: ", len(pos))

# create delaunay triangulations
triangles = fun.create_delaunay_triangles(pos) #create triangles
print("number of triangles found: ", len(triangles))

# define the number of cols and rows of the space
hash_density = 2

# hash the space for faster searches. groups triangles in hash categories
hashed_triangles = fun.hash_triangles(pos, triangles, hash_density) # 2 means divide the space in 2 cols and 2 rows
print("number of hashed triangles found: ", len(hashed_triangles))

# search the rhythm space
#########################

s = (0.5,0.5) # coordinates to search

output_patt = fun.position2pattern(s, all_pattlists,  pos, triangles, hashed_triangles, hash_density)

print("searched for coordinates: ", s)
print ("obtained pattern:", output_patt)

############# plot
#plt.scatter(mds_pos[:,0], mds_pos[:,1], color="0", s=5, marker="o", alpha=0.5, edgecolors="none")
#plt.xlim(-2,2)
#plt.ylim(-2,2)
#plt.title("rhythmspace")
#plt.gca().set_aspect(1)
#plt.show()