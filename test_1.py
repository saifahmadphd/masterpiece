import pandas as pd
from numpy.random import normal
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
from sklearn.neighbors import KernelDensity
from matplotlib.collections import PolyCollection
from sklearn.model_selection import GridSearchCV
#generate a sample
sample=normal(loc=5,scale=2,size=1000)
#plot a historgam of the sampels
plt.figure(1) #Figure 1
plt.hist(sample,bins=10)
#fit distribution
sample_mean=np.mean(sample)
sample_scale=np.std(sample)
sample_dist=stats.norm(loc=sample_mean,scale=sample_scale)
probabilities=[sample_dist.pdf(value) for value in range(-2,12)]
plt.figure(figsize=(10,6))#Figure 2
plt.plot(range(-2,12),probabilities,color='blue', linewidth=2)
plt.xlim(-2,12)
plt.ylim(0,1)
plt.fill_between(range(-2,12),probabilities, color='red',alpha=0.1)
plt.title('Distribution',fontsize=14)
plt.xlabel('x',fontsize=14)
plt.ylabel('pdf(x)',fontsize=14)
plt.tick_params(axis='both',which='major',labelsize=12)
#plt.grid(visible=True)
plt.savefig('distribution.png', dpi=600)
sample_dist.ppf(0.5)#to calculate the percentile of a distribution


#reading excel file for power consumption
power_data23=pd.read_excel('rte_power_data_2023.xls') #data for 2023 available only until sept, no data for feb
jan23=power_data23.iloc[19:50,2:50].to_numpy()
mar23=power_data23.iloc[51:82,2:50].to_numpy()
apr23=power_data23.iloc[83:113,2:50].to_numpy()
may23=power_data23.iloc[114:145,2:50].to_numpy()
jun23=power_data23.iloc[146:176,2:50].to_numpy()
jul23=power_data23.iloc[177:207,2:50].to_numpy()
aug23=power_data23.iloc[208:238,2:50].to_numpy()
sep23=power_data23.iloc[239:268,2:50].to_numpy()
#entire data for 2023
power_data23arr=power_data23.iloc[19:268,2:50].to_numpy()

#power data for 2022
power_data22=pd.read_excel('rte_power_data_2022.xls') #data for 2022, available for the entire year
power22=power_data22.iloc[19:395,2:50].to_numpy() # removing headers and titles

np.nansum(power22)# sum without considering 'nan'
# modeling the distribution of power 
np.nansum(power22/np.nansum(power22))
norm_power22=power22/np.nansum(power22) #normalised power draw 
avg_con_per_house_per_year=4700 #units in x10^2Whr, 35m2=33180, 60m2=47990, 100m2=100270
scale_factor=avg_con_per_house_per_year*2# '2' denotes the inverse of half an hour sample time
house_power22=norm_power22*scale_factor

house_power22=house_power22.astype(float)# convert datatype to float array
house_power22_clean=house_power22[~np.isnan(house_power22[:,0]),:] # select only those rows which dont have 'nan'    

bw=np.linspace(1e-10, 0.1, 100)#Range of the bandwith parameter
best_bw=[]
model=[]
for i in range(len(house_power22_clean[0,:])):
    data=house_power22_clean[:,i].reshape(len(house_power22_clean[:,i]),1)#replace 0 with 'i' in a loop to do the same for each time slot
    #for searching the optimal bandwidth for the KDE fit
#    params = {'bandwidth': bw} #parameter to be optimized
#    grid = GridSearchCV(KernelDensity(), params,cv=len(sample1)) #cv is equal to the no of samples
#    grid.fit(sample1)
#    best_bw_fit=grid.best_index_ #location of best bandwidth from the fit
#    best_bw.append(bw[best_bw_fit]) #selection of best bandwidth from the search space
    model.append(KernelDensity(bandwidth=0.03, kernel='gaussian'))
    model[i].fit(data)
 
ts=3#select the time slot to plot
#plot pdf of distribution
x1 = np.linspace(0,1,1000)
x1=x1.reshape(-1, 1)
probabilities1 = model[ts].score_samples(x1)
probabilities1 = np.exp(probabilities1)
plt.figure(3) #Figure 3
plt.plot(x1,probabilities1)
plt.xlim(0,1)
plt.ylim(0,10)

sum(house_power22_clean)

np.mean(sample)
y1=model[ts].sample(100)
plt.figure(4) #Figure 4
plt.hist(y1,bins=50)#Hist of samples from the model
plt.figure(5) #Figure 5
plt.hist(house_power22_clean[:,ts],bins=40)# Hist of actual data


#house_power22_clean=house_power22_clean[:,0:2]



plt.figure(6)

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


ax = plt.figure().add_subplot(projection='3d')

x = np.linspace(0, 1, 1000)
lambdas = range(len(house_power22_clean[0,:]))

# verts[i] is a list of (x, y) pairs defining polygon i.
gamma = np.vectorize(math.gamma)
verts = [polygon_under_graph(x,  np.exp(model[l].score_samples(x.reshape(-1, 1))))
         for l in lambdas]
colors = []

for i in range(48):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
poly = PolyCollection(verts, facecolors = colors, alpha=.5, edgecolors='k')
ax.add_collection3d(poly, zs=lambdas, zdir='y')

ax.set(xlim=(0, 1), ylim=(0, 47), zlim=(0, 10),
       xlabel='Energy Demand', ylabel=r'Time Slot', zlabel='pdf')


plt.show()


























