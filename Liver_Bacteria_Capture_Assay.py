import numpy as np
import math
import itertools
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import scipy.stats as st
import pandas as pd

# %%

##################################
# Set Parameters for simulation
##################################

bacteria_number = 500  # number of bacteria simulated in each iteration
n_iter = 500  # number of simulations
stickness_increase = 1.2  # multiplicative increase of stickness when going through a cell
stickiness = 0.1
cell_number = 30 # number of macrophages
vessels_number = 4  # number of vessels in first stratum
cell_height = 15 # macrophage height
cell_width = 5 # macrophage width
bacteria_width = 1 # (um)
bacteria_depth = 1 # (um)
walk_width = 1
sinusoid_width = 11  # (um)
sinusoid_length1 = 50  # (um)
sinusoid_length2 = 55  # (um)
sinusoid_length3 = 45  # (um)
sinusoid_length = (sinusoid_length1*2 + sinusoid_length2*2 + sinusoid_length3)

# %%

##################################
# A. To determine Lambda - Visualize exponential distribution
##################################

fig,ax=plt.subplots(figsize=(8,6))
plt.hist(np.random.exponential(500,size=5000),bins=np.linspace(0,1000,100),
        color="blue",alpha=0.7,label=r'$\lambda = 500$',density=True)
plt.hist(np.random.exponential(25,size=5000),bins=np.linspace(0,1000,100),
        color="red",alpha=0.7,label=r'$\lambda = 25$',density=True)
plt.ylabel("Probability", fontsize=24)
plt.xlabel("Y coordinate", fontsize=22)
plt.xlim(0,sinusoid_length1*2+sinusoid_length2*2+sinusoid_length3)
plt.legend(fontsize=22)
plt.show()
# fig.savefig("./Histo_lambda_shape.png", bbox_inches="tight",dpi=300)

fig,ax=plt.subplots(figsize=(8,6))
x_ = np.linspace(0,10000,1000)
plt.plot(x_, (1/500.)*np.exp(-(x_/500.)), color="blue",label=r'$\lambda = 500$',linewidth=3.5)
plt.plot(x_, (1/25.)*np.exp(-(x_/25.)), color="red",label=r'$\lambda = 25$',linewidth=3.5)
plt.xlim(0,sinusoid_length1*2+sinusoid_length2*2+sinusoid_length3)
plt.legend(fontsize=22)
plt.ylabel("Probability of KC location",fontsize=24)
plt.xticks([0,255],["PR", "CV"], fontsize=22)
plt.show()
# fig.savefig("./Exponential_distribution_lambda.png", bbox_inches="tight",dpi=300)

# %%
perc_stopped_mean = []
perc_stopped_std = []
distance_max_mean = []
distance_max_std = []
for stickiness in np.linspace(0,1,num=21):
    for lambda_exp in np.linspace(10,500,num=21):
        perc_stopped = []

        ##################################
        # B. START of forward loop
        ##################################
        for i in range(n_iter):

            # Initiate the array with zeros -> simulates sinusoidal space
            array1 = np.zeros((vessels_number, sinusoid_length1, sinusoid_width))
            array2 = np.zeros((vessels_number * 2, sinusoid_length2, sinusoid_width))
            array3 = np.zeros((vessels_number * 4, sinusoid_length3, sinusoid_width))
            array4 = np.zeros((vessels_number * 2, sinusoid_length2, sinusoid_width))
            array5 = np.zeros((vessels_number, sinusoid_length1, sinusoid_width))

            ##################################
            # C. Create network and place macrophages:
            ##################################

            y_coord_c = []
            x_coord_c = []
            vessel_c = []
            stratum_c = []
            sinusoid_length = (sinusoid_length1*2 + sinusoid_length2*2 + sinusoid_length3)
            while len(y_coord_c) < cell_number:
                # give a 50/50 chance of each cell to stick to the left or the right
                if np.random.rand() < 0.5:
                    x_coord_c_i = 0
                else:
                    x_coord_c_i = sinusoid_width - cell_width
                x_coord_c.append(x_coord_c_i)
                # Select y coordinate of macrophage between 0 and sinusoid_length-cell_height+1
                y_coord_c_i = 300
                while ((y_coord_c_i > sinusoid_length - cell_height + 1) | (y_coord_c_i < 0)):
                    y_coord_c_i = int((1-(np.random.exponential(lambda_exp)/(sinusoid_length-cell_height+1)))*(sinusoid_length-cell_height+1))
                    # exclude cells on forks
                    h1 = ((y_coord_c_i>=sinusoid_length1 - cell_height + 1) & (y_coord_c_i < sinusoid_length1))
                    h2 = ((y_coord_c_i>=sinusoid_length1+sinusoid_length2 - cell_height + 1) & (y_coord_c_i < sinusoid_length1+sinusoid_length2))
                    h3 = ((y_coord_c_i >= sinusoid_length1+sinusoid_length2+sinusoid_length3 - cell_height + 1) & (y_coord_c_i < sinusoid_length1+sinusoid_length2+sinusoid_length3))
                    h4 = ((y_coord_c_i >= sinusoid_length1+sinusoid_length2*2+sinusoid_length3 - cell_height + 1) & (y_coord_c_i < sinusoid_length1+sinusoid_length2*2+sinusoid_length3))
                if (not ((h1 or h2) or (h3 or h4))):
                    y_coord_c.append(y_coord_c_i)
                else:
                    del x_coord_c[-1]
                    continue

                # Select stratum level, determine cell overlap, assign position
                # Select tree depth based on the y_coord_c_i
                # Stratum 5 (bottom)
                if (y_coord_c_i < sinusoid_length1 - cell_height + 1):
                    # chose between vessels at this tree depth
                    vessel_i = np.random.randint(0, vessels_number)
                    vessel_c.append(vessel_i)
                    stratum_c.append(5)
                    y_min = y_coord_c_i
                    y_max = y_min + cell_height
                    # Check overlap
                    if (sum(sum(array5[vessel_i][y_min:y_max, x_coord_c_i : x_coord_c_i + cell_width]))) > 0:
                        del x_coord_c[-1]
                        del y_coord_c[-1]
                        del stratum_c[-1]
                        del vessel_c[-1]
                    # Fill with 1
                    else:
                        for l in range(cell_width):
                            for m in range(cell_height):
                                array5[vessel_i][y_min + m, x_coord_c_i + l] = 1  # put 1 where the macrophage stands
                # Stratum 4
                if ((y_coord_c_i >= sinusoid_length1) & (y_coord_c_i < sinusoid_length1+sinusoid_length2 - cell_height + 1)):
                    vessel_i = np.random.randint(0, vessels_number*2)
                    vessel_c.append(vessel_i)
                    stratum_c.append(4)
                    y_min = y_coord_c_i-(sinusoid_length1)
                    y_max = y_min + cell_height
                    # Check overlap
                    if (sum(sum(array4[vessel_i][y_min:y_max, x_coord_c_i : x_coord_c_i + cell_width]))) > 0:
                        del x_coord_c[-1]
                        del y_coord_c[-1]
                        del stratum_c[-1]
                        del vessel_c[-1]
                    # Fill with 1
                    else:
                        for l in range(cell_width):
                            for m in range(cell_height):
                                array4[vessel_i][y_min + m, x_coord_c_i + l] = 1
                # Stratum 3
                if ((y_coord_c_i >= sinusoid_length1+sinusoid_length2) & (y_coord_c_i < sinusoid_length1+sinusoid_length2+sinusoid_length3 - cell_height + 1)):
                    vessel_i = np.random.randint(0, vessels_number*4)
                    vessel_c.append(vessel_i)
                    stratum_c.append(3)
                    y_min = y_coord_c_i-(sinusoid_length1+sinusoid_length2)
                    y_max = y_min + cell_height
                    # Check overlap
                    if (sum(sum(array3[vessel_i][y_min:y_max, x_coord_c_i : x_coord_c_i + cell_width]))) > 0:
                        del x_coord_c[-1]
                        del y_coord_c[-1]
                        del stratum_c[-1]
                        del vessel_c[-1]
                    # Fill with 1
                    else:
                        for l in range(cell_width):
                            for m in range(cell_height):
                                array3[vessel_i][y_min + m, x_coord_c_i + l] = 1
                # Stratum 2
                if ((y_coord_c_i >= sinusoid_length1+sinusoid_length2+sinusoid_length3) & (y_coord_c_i < sinusoid_length1+sinusoid_length2*2+sinusoid_length3 - cell_height + 1)):
                    vessel_i = np.random.randint(0, vessels_number*2)
                    vessel_c.append(vessel_i)
                    stratum_c.append(2)
                    y_min = y_coord_c_i-(sinusoid_length1+sinusoid_length2+sinusoid_length3)
                    y_max = y_min + cell_height
                    # Check overlap
                    if (sum(sum(array2[vessel_i][y_min:y_max, x_coord_c_i : x_coord_c_i + cell_width]))) > 0:
                        del x_coord_c[-1]
                        del y_coord_c[-1]
                        del stratum_c[-1]
                        del vessel_c[-1]
                    # Fill with 1
                    else:
                        for l in range(cell_width):
                            for m in range(cell_height):
                                array2[vessel_i][y_min + m, x_coord_c_i+l] = 1
                # Stratum 1 (top)
                if ((y_coord_c_i >= sinusoid_length1+sinusoid_length2*2+sinusoid_length3) & (y_coord_c_i < sinusoid_length1*2+sinusoid_length2*2+sinusoid_length3 - cell_height + 1)):
                    vessel_i = np.random.randint(0, vessels_number)
                    vessel_c.append(vessel_i)
                    stratum_c.append(1)
                    y_min = y_coord_c_i-(sinusoid_length1+sinusoid_length2*2+sinusoid_length3)
                    y_max = y_min + cell_height
                    # Check overlap
                    if (sum(sum(array1[vessel_i][y_min:y_max, x_coord_c_i : x_coord_c_i + cell_width]))) > 0:
                        del x_coord_c[-1]
                        del y_coord_c[-1]
                        del stratum_c[-1]
                        del vessel_c[-1]
                    # Fill with 1
                    else:
                        for l in range(cell_width):
                            for m in range(cell_height):
                                array1[vessel_i][y_min + m, x_coord_c_i + l] = 1

            ##################################
            # D. Merge of vessels at radomn
            ##################################

            # Randomly merge vessels - stratum 4
            paired_vessel1 = []
            pair_coupled = []

            pair_list = list(itertools.combinations(range(vessels_number * 4), 2))
            # Calculate number_coupple = n*(n-1)/2 and find n:
            # (np.sqrt(1+len(pair_list)*8)+1)/2
            while (len(pair_list) > 0):
                pair_i = np.random.choice(range(int((np.sqrt(1+len(pair_list)*8)+1)/2 - 1)), size=1)
                paired_vessel1.append(pair_list[:int((np.sqrt(1+len(pair_list)*8)+1)/2 - 1)][pair_i[0]])
                pair_coupled.extend(list(pair_list[:vessels_number * 4 - 1][pair_i[0]]))
                pair_list = [pair_list[i] for i in range(len(pair_list)) if len(list(set(pair_list[i]).intersection(set(pair_coupled)))) == 0]


            # Randomly merge vessels - stratum 5
            paired_vessel2 = []
            pair_coupled = []
            pair_list = list(itertools.combinations(range(vessels_number * 2), 2))
            # Calculate number_coupple = n*(n-1)/2 and find n:
            # (np.sqrt(1+len(pair_list)*8)+1)/2
            while (len(pair_list) > 0):
                pair_i = np.random.choice(range(int((np.sqrt(1+len(pair_list)*8)+1)/2 - 1)), size=1)
                paired_vessel2.append(pair_list[:int((np.sqrt(1+len(pair_list)*8)+1)/2 - 1)][pair_i[0]])
                pair_coupled.extend(list(pair_list[:vessels_number * 4 - 1][pair_i[0]]))
                pair_list = [pair_list[i] for i in range(len(pair_list)) if len(list(set(pair_list[i]).intersection(set(pair_coupled)))) == 0]

            ##################################
            # E. Bacteria dynamics
            ##################################

            # b) Triangular "distribution"
            if (sinusoid_width % 2 == 0):
                triang_dist = list(np.linspace(0.25, 1, int(math.ceil(sinusoid_width/2.))+1))
                triang_dist = triang_dist[::-1] + triang_dist[1:]
                # plt.plot(triang_dist)
            else:
                triang_dist = list(np.linspace(0.25, 1, int(math.ceil(sinusoid_width/2.))))
                triang_dist = triang_dist[::-1] + triang_dist
                # plt.plot(triang_dist)


            x_coord_b = []
            vessel_b_i = []
            vessel_b_ii = []
            vessel_b_iii = []
            vessel_b_iv = []
            vessel_b_v = []
            bac_status = []
            success = 0
            for b in range(bacteria_number):
                # Get x_coord (which is the next x position of the bacteria?)
                x_coord_b.append(np.random.randint(0, int(sinusoid_width - bacteria_width + 1)))  # starting point for bacteria b
                for w in range(sinusoid_length - 1):
                    new_x = x_coord_b[w+(b*(sinusoid_length))]+int(round(np.random.normal(0, walk_width)*(triang_dist[x_coord_b[w+(b*(sinusoid_length))]])))
                    # to bounce off the sinusoidal edges
                    if new_x < 0:
                        x_coord_b.append(abs(new_x))
                    elif new_x > sinusoid_width - 1:
                        x_coord_b.append(sinusoid_width-1-(new_x-(sinusoid_width-1)))
                    else:
                        x_coord_b.append(new_x)
                # Get vessels path (which vessel?)
                # Select vessel in stratum 1
                vessel_i = np.random.randint(0, vessels_number)  # 0-3
                # Select vessel in stratum 2 (between 2 from vessel_i)
                vessel_ii = np.random.randint(vessel_i * 2, vessel_i * 2 + 2)  # 0-7
                # Select vessel in stratum 3 (between 2 from vessel_ii)
                vessel_iii = np.random.randint(vessel_ii * 2, vessel_ii * 2 + 2)  # 0-15
                # Select vessel in stratum 4 (merged)
                ind1 = [i for i in range(len(paired_vessel1)) if vessel_iii in paired_vessel1[i]][0]
                vessel_iv = paired_vessel1[ind1]
                # Select vessel in stratum 5 (merged)
                ind2 = [i for i in range(len(paired_vessel2)) if ind1 in paired_vessel2[i]][0]
                vessel_v = paired_vessel2[ind2]
                vessel_b_i.append(vessel_i)
                vessel_b_ii.append(vessel_ii)
                vessel_b_iii.append(vessel_iii)
                vessel_b_iv.append(ind1)
                vessel_b_v.append(ind2)

                # for each bacterium, assess each step in its walk and see if it is allowed to progress
                bac_status.append(1)
                # stratum 1 (top)
                sinusoid_length_i = sinusoid_length1
                y_sub_i = 0
                bact_in_cell = [0]
                stickiness_c = stickiness
                for r in range(sinusoid_length_i):
                    if array1[vessel_i][sinusoid_length_i-1-r, x_coord_b[(b*sinusoid_length)+y_sub_i+r]] == 1.0 and bac_status[-1] == 1:
                        if bact_in_cell[-1] == 1:  # if bacteria was on a cell in previous step, increase stickness
                            stickiness_c = stickiness_c*stickness_increase
                        bact_in_cell.append(1)  # add 1 if bacteria is on a cell
                        if np.random.rand() > (1.-stickiness_c):
                            bac_status.append(0)
                        else:
                            bac_status.append(1)
                    elif bac_status[-1] == 0:
                        bac_status.append(0)
                    else:
                        bac_status.append(1)
                        bact_in_cell.append(0)  # add 0 if bacteria is not on a cell
                        stickiness_c = stickiness  # return to original stickness if bacteria is not on a cell

                # stratum 2
                sinusoid_length_i = sinusoid_length2
                y_sub_i = sinusoid_length1
                bact_in_cell = [0]
                stickiness_c = stickiness
                for r in range(sinusoid_length_i):
                    if array2[vessel_ii][sinusoid_length_i-1-r, x_coord_b[(b*sinusoid_length)+y_sub_i+r]] == 1.0 and bac_status[-1] == 1:
                        if bact_in_cell[-1] == 1:  # if bacteria was on a cell in previous step, increase stickness
                            stickiness_c = stickiness_c*stickness_increase
                        bact_in_cell.append(1)  # add 1 if bacteria is on a cell
                        if np.random.rand() > (1.-stickiness_c):
                            bac_status.append(0)
                        else:
                            bac_status.append(1)
                    elif bac_status[-1] == 0:
                        bac_status.append(0)
                    else:
                        bac_status.append(1)
                        bact_in_cell.append(0)  # add 0 if bacteria is not on a cell
                        stickiness_c = stickiness  # return to original stickness if bacteria is not on a cell
                # stratum 3
                sinusoid_length_i = sinusoid_length3
                y_sub_i = sinusoid_length1 + sinusoid_length2
                bact_in_cell = [0]
                stickiness_c = stickiness
                for r in range(sinusoid_length_i):
                    if array3[vessel_iii][sinusoid_length_i-1-r, x_coord_b[(b*sinusoid_length)+y_sub_i+r]] == 1.0 and bac_status[-1] == 1:
                        if bact_in_cell[-1] == 1:  # if bacteria was on a cell in previous step, increase stickness
                            stickiness_c = stickiness_c*stickness_increase
                        bact_in_cell.append(1)  # add 1 if bacteria is on a cell
                        if np.random.rand() > (1.-stickiness_c):
                            bac_status.append(0)
                        else:
                            bac_status.append(1)
                    elif bac_status[-1] == 0:
                        bac_status.append(0)
                    else:
                        bac_status.append(1)
                        bact_in_cell.append(0)  # add 0 if bacteria is not on a cell
                        stickiness_c = stickiness  # return to original stickness if bacteria is not on a cell
                # stratum 4
                sinusoid_length_i = sinusoid_length2
                y_sub_i = sinusoid_length1 + sinusoid_length2 + sinusoid_length3
                bact_in_cell = [0]
                stickiness_c = stickiness
                for r in range(sinusoid_length_i):
                    if array4[ind1][sinusoid_length_i-1-r, x_coord_b[(b*sinusoid_length)+y_sub_i+r]] == 1.0 and bac_status[-1] == 1:
                        if bact_in_cell[-1] == 1:  # if bacteria was on a cell in previous step, increase stickness
                            stickiness_c = stickiness_c*stickness_increase
                        bact_in_cell.append(1)  # add 1 if bacteria is on a cell
                        if np.random.rand() > (1.-stickiness_c):
                            bac_status.append(0)
                        else:
                            bac_status.append(1)
                    elif bac_status[-1] == 0:
                        bac_status.append(0)
                    else:
                        bac_status.append(1)
                        bact_in_cell.append(0)  # add 0 if bacteria is not on a cell
                        stickiness_c = stickiness  # return to original stickness if bacteria is not on a cell
                # stratum 5
                sinusoid_length_i = sinusoid_length1
                y_sub_i = sinusoid_length1 + sinusoid_length2*2 + sinusoid_length3
                bact_in_cell = [0]
                stickiness_c = stickiness
                for r in range(sinusoid_length_i-1):
                    if array5[ind2][sinusoid_length_i-1-r, x_coord_b[(b*sinusoid_length)+y_sub_i+r]] == 1.0 and bac_status[-1] == 1:
                        if bact_in_cell[-1] == 1:  # if bacteria was on a cell in previous step, increase stickness
                            stickiness_c = stickiness_c*stickness_increase
                        bact_in_cell.append(1)  # add 1 if bacteria is on a cell
                        if np.random.rand() > (1.-stickiness_c):
                            bac_status.append(0)
                        else:
                            bac_status.append(1)
                    elif bac_status[-1] == 0:
                        bac_status.append(0)
                    else:
                        bac_status.append(1)
                        bact_in_cell.append(0)  # add 0 if bacteria is not on a cell
                        stickiness_c = stickiness  # return to original stickness if bacteria is not on a cell

                if bac_status[-1] == 1: # this means the bacterium made it to the end
                    success +=1
            perc_stopped.append(np.round(float(bacteria_number - success)/float(bacteria_number),4)*100) # % of bacteria stopped in this iteration
        perc_stopped_mean.append(np.mean(perc_stopped))
        perc_stopped_std.append(np.std(perc_stopped))
        # Maximum distance reached by bacteria
        distance_max = []
        for bac_i in range(bacteria_number):
            bac_status_i = bac_status[bac_i*sinusoid_length:(bac_i+1)*sinusoid_length]
            distance_max.append(sum([i==1 for i in bac_status_i]))
        distance_max_mean.append(np.mean(distance_max))
        distance_max_std.append(np.std(distance_max))

# %%

################################
# Convert results to DataFrame:
################################
perc_stopped_mean_df = pd.DataFrame([perc_stopped_mean[i*21:(i+1)*21] for i in range(21)])
perc_stopped_mean_df["stickness"] = ["stickness_"+"%.2f" % i for i in np.linspace(0,1,21)]
perc_stopped_mean_df = perc_stopped_mean_df.set_index("stickness")
perc_stopped_mean_df.columns = ["lambda_"+"%.2f" %i for i in np.linspace(10,500,21)]

perc_stopped_std_df = pd.DataFrame([perc_stopped_std[i*21:(i+1)*21] for i in range(21)])
perc_stopped_std_df["stickness"] = ["stickness_"+"%.2f" % i for i in np.linspace(0,1,21)]
perc_stopped_std_df = perc_stopped_std_df.set_index("stickness")
perc_stopped_std_df.columns = ["lambda_"+"%.2f" %i for i in np.linspace(10,500,21)]

distance_max_mean_df = pd.DataFrame([distance_max_mean[i*21:(i+1)*21] for i in range(21)])
distance_max_mean_df["stickness"] = ["stickness_"+"%.2f" % i for i in np.linspace(0,1,21)]
distance_max_mean_df = distance_max_mean_df.set_index("stickness")
distance_max_mean_df.columns = ["lambda_"+"%.2f" %i for i in np.linspace(10,500,21)]

distance_max_std_df = pd.DataFrame([distance_max_std[i*21:(i+1)*21] for i in range(21)])
distance_max_std_df["stickness"] = ["stickness_"+"%.2f" % i for i in np.linspace(0,1,21)]
distance_max_std_df = distance_max_std_df.set_index("stickness")
distance_max_std_df.columns = ["lambda_"+"%.2f" %i for i in np.linspace(10,500,21)]

# # Save results
# distance_max_mean_df.to_csv("./distance_max_mean_df.csv")
# distance_max_std_df.to_csv("./distance_max_std_df.csv")
# perc_stopped_mean_df.to_csv("./perc_stopped_mean_df.csv")
# perc_stopped_std_df.to_csv("./perc_stopped_std_df.csv")

# %%
#######################
# Visualize results
#######################

# # Read results
# results_folder = "."
# distance_max_mean_df = pd.read_csv(results_folder + "distance_max_mean_df.csv", index_col="stickness")
# perc_stopped_mean_df = pd.read_csv(results_folder + "perc_stopped_mean_df.csv", index_col="stickness")
# distance_max_mean_df = pd.read_csv(results_folder + "distance_max_mean_df.csv", index_col="stickness")
# perc_stopped_mean_df = pd.read_csv(results_folder + "perc_stopped_mean_df.csv", index_col="stickness")
#%%
import seaborn as sn
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# %%
distance_max_mean_df.columns = [re.sub("_",": ",i) for i in distance_max_mean_df.columns]
distance_max_mean_df = distance_max_mean_df.set_index(pd.Series([re.sub("_",": ",i) for i in distance_max_mean_df.index]))
perc_stopped_mean_df.columns = [re.sub("_",": ",i) for i in perc_stopped_mean_df.columns]
perc_stopped_mean_df = perc_stopped_mean_df.set_index(pd.Series([re.sub("_",": ",i) for i in perc_stopped_mean_df.index]))
# Heat map:

fig, ax = plt.subplots(figsize=(9,8))
sn.heatmap(distance_max_mean_df[1:])
plt.show()
# fig.savefig(results_folder + "Heatmap_distance_max_mean_df_NoZero.png", bbox_inches="tight",dpi=200)
# %%
cmap = plt.cm.get_cmap('gnuplot')
color_list = [cmap(i) for i in np.linspace(0,1,21)]
fig, ax = plt.subplots(figsize=(9,8))
k=0
for i in perc_stopped_mean_df.index[1:]:
    plt.plot(perc_stopped_mean_df.loc[i,], color=color_list[k], label=i)
    k+=1
plt.legend(bbox_to_anchor=(1,1.015),fontsize=14)
plt.ylabel("Percentage of stopped", fontsize=14)
plt.xticks(rotation="vertical")
plt.show()
# fig.savefig(results_folder + "Plot_per_stickness_perc_stopped_mean_df_NoZero.png", bbox_inches="tight",dpi=200)

# %%

fig, ax = plt.subplots(figsize=(9,8))
k=0
for i in perc_stopped_mean_df.columns:
    plt.plot(perc_stopped_mean_df.loc[perc_stopped_mean_df.index[1:],i], color=color_list[k], label=i)
    k+=1
plt.legend(bbox_to_anchor=(1,1.015),fontsize=14)
plt.ylabel("Percentage of stopped", fontsize=14)
plt.xticks(rotation="vertical")
plt.show()
# fig.savefig(results_folder + "Plot_per_lambda_perc_stopped_mean_df_NoZero.png", bbox_inches="tight",dpi=200)
# %%
fig, ax = plt.subplots(figsize=(9,8))
k=0
for i in distance_max_mean_df.index[1:]:
    plt.plot(distance_max_mean_df.loc[i,], color=color_list[k], label=i)
    k+=1
plt.legend(bbox_to_anchor=(1,1.015),fontsize=14)
plt.ylabel("Maximum distance", fontsize=14)
plt.xticks(rotation="vertical")
plt.show()
# fig.savefig(results_folder + "Plot_per_stickness_distance_max_mean_df_NoZero.png", bbox_inches="tight",dpi=200)

# %%
fig, ax = plt.subplots(figsize=(9,8))
k=0
for i in distance_max_mean_df.columns:
    plt.plot(distance_max_mean_df.loc[distance_max_mean_df.index[1:],i], color=color_list[k], label=i)
    k+=1
plt.legend(bbox_to_anchor=(1,1.015),fontsize=14)
plt.ylabel("Maximum distance", fontsize=14)
plt.xticks(rotation="vertical")
plt.show()
# fig.savefig(results_folder + "Plot_per_lambda_Maximum distance_df_NoZero.png", bbox_inches="tight",dpi=200)

# %%
fig, ax = plt.subplots(figsize=(9,8))
sn.heatmap(perc_stopped_mean_df[1:]) # without first element
plt.show()
# fig.savefig(results_folder + "Heatmap_perc_stopped_mean_df_NoZero.png", bbox_inches="tight",dpi=200)

# %%
#
# #####################
# # Plot the network and the  bacteria track
# #####################
bac_path_on = True
if bacteria_number<=10:  # only if less than 10 bugs you will visualize bacteria tracts
    bac_path_on = True
else:
    bac_path_on = False

vessels_b = [vessel_b_i, vessel_b_ii, vessel_b_iii, vessel_b_iv, vessel_b_v]
num_levels = 5
num_plots=32
colours = ["orchid", "mediumseagreen","dodgerblue",  "gold",
            "sienna","darkcyan", "crimson", "green","darkorange", "gray"]# [list(np.random.rand(3)) for i in range(bacteria_number)]
fig =plt.figure(figsize=(40,40))
fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
X = [(num_levels,num_plots,(4,5), 1, 0),(num_levels,num_plots,(12,13),1,1), (num_levels,num_plots,(20,21),1,2), (num_levels,num_plots,(28,29),1,3),

     (num_levels,num_plots,(34,35),2,0), (num_levels,num_plots,(38,39),2,1), (num_levels,num_plots,(42,43),2,2), (num_levels,num_plots,(46,47),2,3),
     (num_levels,num_plots,(50,51),2,4), (num_levels,num_plots,(54,55),2,5), (num_levels,num_plots,(58,59),2,6), (num_levels,num_plots,(62,63),2,7),

     (num_levels,num_plots,(65,66),3,0), (num_levels,num_plots,(67,68),3,1), (num_levels,num_plots,(69,70),3,2), (num_levels,num_plots,(71,72),3,3),
     (num_levels,num_plots,(73,74),3,4), (num_levels,num_plots,(75,76),3,5), (num_levels,num_plots,(77,78),3,6), (num_levels,num_plots,(79,80),3,7),
     (num_levels,num_plots,(81,82),3,8), (num_levels,num_plots,(83,84),3,9), (num_levels,num_plots,(85,86),3,10), (num_levels,num_plots,(87,88),3,11),
     (num_levels,num_plots,(89,90),3,12), (num_levels,num_plots,(91,92),3,13), (num_levels,num_plots,(93,94),3,14), (num_levels,num_plots,(95,96),3,15),

     (num_levels,num_plots,(98,99),4,0), (num_levels,num_plots,(102,103),4,1), (num_levels,num_plots,(106,107),4,2), (num_levels,num_plots,(110,111),4,3),
     (num_levels,num_plots,(114,115),4,4), (num_levels,num_plots,(118,119),4,5), (num_levels,num_plots,(122,123),4,6), (num_levels,num_plots,(126,127),4,7),

     (num_levels,num_plots,(132,133),5,0),(num_levels,num_plots,(140,141),5,1), (num_levels,num_plots,(148,149),5,2), (num_levels,num_plots,(156,157),5,3)]

for nrows, ncols, plot_number, stratum_i, vessel_x in X:
    sub = fig.add_subplot(nrows, ncols, plot_number)
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_title("Level:"+str(stratum_i)+"; Vessel:"+str(vessel_x))

    if stratum_i == 5:
        sinusoid_length_i = sinusoid_length1
        y_sub_i = sinusoid_length1+sinusoid_length2*2+sinusoid_length3
        y_sup_i = 0
    if stratum_i == 4:
        sinusoid_length_i = sinusoid_length2
        y_sub_i = sinusoid_length1+sinusoid_length2+sinusoid_length3
        y_sup_i = sinusoid_length1
    if stratum_i == 3:
        sinusoid_length_i = sinusoid_length3
        y_sub_i = sinusoid_length1+sinusoid_length2
        y_sup_i = sinusoid_length1+sinusoid_length2
    if stratum_i == 2:
        sinusoid_length_i = sinusoid_length2
        y_sub_i = sinusoid_length1
        y_sup_i = sinusoid_length1+sinusoid_length2+sinusoid_length3
    if stratum_i == 1:
        sinusoid_length_i = sinusoid_length1
        y_sub_i = 0
        y_sup_i = sinusoid_length1+sinusoid_length2*2+sinusoid_length3

    sub.set_xlim(0,sinusoid_width)
    sub.set_ylim(0,(max(sinusoid_length1, sinusoid_length2,sinusoid_length3)))
    sub.axhline(y=sinusoid_length_i, color='k', linestyle='-', linewidth=1)
    sub.axhline(y=0, color='k', linestyle='-', linewidth=1)
    sub.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    sub.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    sub.grid()
    # plot the bacteria (if requested) and the cells
    if bac_path_on:
        for bact in range(bacteria_number):
            colour = colours[bact]  # bact=1
            b_min = bact*sinusoid_length+y_sub_i
            b_max = bact*sinusoid_length+sinusoid_length_i+y_sub_i

            for b in range(b_min, b_max): # stratum_i=1
                if vessels_b[stratum_i-1][bact] == vessel_x:
                    if bac_status[b] == 1:
                        # if b % (sinusoid_length_i+y_sub_i) == 0:
                        #    colour = colours[bact]
                        sub.add_patch(Rectangle((x_coord_b[b],sinusoid_length_i-1-(b-b_min)),bacteria_width,bacteria_depth,facecolor=colour, clip_on=False, linewidth=1, edgecolor='k', alpha=0.7, zorder=50))
    # plot the cells
    for c in range(cell_number):
        if ((stratum_c[c]==stratum_i) & (vessel_c[c]==vessel_x)):
            sub.add_patch(Rectangle((x_coord_c[c], (y_coord_c[c])-y_sup_i), cell_width, cell_height, facecolor='red', linewidth=1, edgecolor='k',alpha=0.7, zorder=40))

plt.show()
# # %%
# file = open("./merged_vessels.txt","w")
# file.write("Stratum 3 to 4:")
# file.write(str(paired_vessel1))
# file.write("Stratum 4 to 5:")
# file.write(str(paired_vessel2))
# file.close()
# fig.savefig("./figure_path_lambda"+str(lambda_exp)+".png", bbox_inches="tight",dpi=300)



# %%
