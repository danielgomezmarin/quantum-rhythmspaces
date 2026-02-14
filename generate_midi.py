# This script generates a MIDI file from a path in the rhythmspace. It uses the functions in functions.py to create the MIDI file. The path is generated using the script generate_path_and_animation.py as a txt file.

"""--- IMPORT LIBRARIES ---"""
import os
import functions as fun
import descriptors as desc
import matplotlib.pyplot as plt
import numpy as np
from qufunctions import txt2path


"""--- IMPORT THE RHYTHMSPACE PATTERN LISTS AND DESCRIPTORS ---"""
# Import all_pattlists (pattern lists), all_names (pattern names), the descriptors and the positions from the rhythmspace from txt files. These txt files were extracted using the script examples.py.
filename = "all_pattlists.txt"
with open(filename  , 'r') as f:
    all_pattlists = eval(f.read())
filename = "all_names.txt"
with open(filename  , 'r') as f:
    all_names = f.readlines()
all_names = [x.strip() for x in all_names] # strip removes the \n at the end of each line

d = []
filename = "descriptors.txt"
with open(filename , 'r') as f:
    for line in f:
        d.append(np.array([float(x) for x in line.split()]))
# transform d from list to numpy array
d = np.array(d)

# import the positions from a txt file
pos = []
filename = "positions.txt"
with open(filename , 'r') as f:
    for line in f:
        pos.append([float(x) for x in line.split()])
# transform pos from list to numpy array
pos = np.array(pos)

"""--- OPTIONAL: PRINT SOME INFO ABOUT THE IMPORTED DATA ---"""

# COMPUTE NUMBER OF INSTRUMENTS
# compute the number of different MIDI values that appear in all_pattlists (this is the number of instruments) and give them as a list
possible_instruments = []
for p in all_pattlists:
    for x in p:
        for y in x:
            if y not in possible_instruments:
                possible_instruments.append(y)

print("number of different instruments found: ", len(possible_instruments))
print("instruments found: ", possible_instruments)

# COMPUTE THE DISTANCE BETWEEN ALL POINTS IN THE RHYTHMSPACE
# compute distances between all pos points with a for loop, store them and create a histogram to evaluate the distribution of distances of the points in the rhythmspace
dists = []
for i in range(len(pos)):
    for j in range(i+1, len(pos)):
        dists.append(np.sqrt((pos[j][1]-pos[i][1])**2 + (pos[j][0]-pos[i][0])**2))
#print("number of distances found: ", len(dists))

# filter out distances > 0.9 and < 0.0
dists = [x for x in dists if x < 0.9 and x > 0.0]

plt.hist(dists, bins=100)
plt.title("Distances between points in the rhythmspace")
plt.xlabel("distance")
plt.ylabel("frequency")
#plt.show()

# choose a random number between 0 and len(pos) to search for a pattern
import random
a = pos[random.randint(0,len(pos)-1)]
def distance(a,b):
    return np.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)

# search for points closer than 0.05 to a
close_points = [x for x in pos if distance(a,x) < 0.1]

# plot them as 2d points
plt.scatter(a[0], a[1], color="r", s=5, marker="o", alpha=0.5, edgecolors="none")
plt.scatter([x[0] for x in close_points], [x[1] for x in close_points], color="b", s=5, marker="o", alpha=0.5, edgecolors="none")
plt.gca().set_aspect(1)
#plt.show()

# save plot as svg
#plt.savefig("close_points.svg")

#print x and y range in pos
print("x range: ", min(pos[:,0]), max(pos[:,0]))
print("y range: ", min(pos[:,1]), max(pos[:,1]))
pos_max_x = max(pos[:,0])
pos_min_x = min(pos[:,0])
pos_max_y = max(pos[:,1])
pos_min_y = min(pos[:,1])

"""--- PREPARE THE RHYTHMSPACE ---"""
# create delaunay triangulations
triangles = fun.create_delaunay_triangles(pos) #create triangles
#print("number of triangles found: ", len(triangles))

# define the number of cols and rows of the space
hash_density = 2

# hash the space for faster searches. groups triangles in hash categories
hashed_triangles = fun.hash_triangles(pos, triangles, hash_density) # 2 means divide the space in 2 cols and 2 rows
#print("number of hashed triangles found: ", len(hashed_triangles))


# IMPORT THE QUANTUM RANDOM WALK PATH
# import path from a txt file
filename = "new_path_gaussian.txt"
path = txt2path(filename)


"""#print max and min values of the path in both axes
path_max_x = max([x[0] for x in path])
path_min_x = min([x[0] for x in path])
path_max_y = max([x[1] for x in path])
path_min_y = min([x[1] for x in path])

#Affine transform manually path to have same min and max values as pos array
for i in range(len(path)):
    path_new = [0,0]
    path_new[0] = (path[i][0] - path_min_x) * (pos_max_x - pos_min_x) / (path_max_x - path_min_x) + pos_min_x
    path_new[1] = (path[i][1] - path_min_y) * (pos_max_y - pos_min_y) / (path_max_y - path_min_y) + pos_min_y
    path[i] = path_new"""


"""---RESAMPLE THE PATH AND CREATE THE MIDI FILE USING THE FUNCTIONS IN functions.py ---"""
# resample the path
resampled_path = fun.path_resampling(path, 16*8)
print("resampled path: ", resampled_path)

# resample path to pattern
final_pattern = fun.resampled_path2pattern(resampled_path, all_pattlists, pos, triangles, hashed_triangles, hash_density)

# pattern to MIDI 
fun.pattern2midipattern(final_pattern, "gaussian_potential.mid")




