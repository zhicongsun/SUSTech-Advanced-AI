import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import animation
from PIL import Image
import matplotlib.patches as mpatches
from math import pi
from numpy import cos, sin
import pandas as pd

def cal_global_center(sites):
    sum_x = 0
    sum_y = 0
    cnt_sites = 0
    for i in range(len(sites)):
            cnt_sites = cnt_sites+1
            sum_x = sum_x + sites[i].x_location
            sum_y = sum_y + sites[i].y_location
    x_location = sum_x/cnt_sites
    y_location = sum_y/cnt_sites
    return x_location,y_location

def cal_nextcenter(density,distance_matrix,centers,pre_siteid_of_center):
    distance_all_sites = []
    n_sites = len(density)
    n_centers = len(centers)
    for i in range(n_sites):
        total_distance = 1
        for j in range(n_centers):
            total_distance = total_distance * distance_matrix[i][pre_siteid_of_center[j]]
        distance_all_sites.append(total_distance)
        if density[i] == 0:
            distance_all_sites[i] = 0
    next_center_id = distance_all_sites.index(max(distance_all_sites))
    return next_center_id

def cal_distance(sites):
    distance_list = []
    n_sites = len(sites)
    distance_matrix = np.zeros((n_sites,n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            distance_matrix[i][j] = ( ((sites[i].x_location - sites[j].x_location) **2)+((sites[i].y_location - sites[j].y_location) **2) ) **0.5
            distance_matrix[j][i] = distance_matrix[i][j]
            distance_list.append(distance_matrix[i][j])
            print((i*n_sites+j)/(n_sites*n_sites))
    return distance_matrix,distance_list

def cal_density(sites,distance_matrix,radius):
    density = []
    n_sites = len(sites)
    for i in range(n_sites):
        cnt_site = 0
        for j in range(n_sites):
            if distance_matrix[i][j] < radius:
                cnt_site = cnt_site + 1
        density.append(cnt_site)
    return density


def assign_center(centers,site):
    min_distance = [( ((site.x_location-centers[0].x_location) ** 2) + ((site.y_location-centers[0].y_location)**2) ) ** 0.5,0]
    for i in range(len(centers)):
        distance = ( ((site.x_location-centers[i].x_location) ** 2) + ((site.y_location-centers[i].y_location) **2) ) ** 0.5
        if distance < min_distance[0]:
            min_distance[0] = distance
            min_distance[1] = centers[i].id
    return centers[min_distance[1]]

def cal_center(id,sites):
    center = Center(id,0,0)
    sum_x = 0
    sum_y = 0
    cnt_sites = 0
    for i in range(len(sites)):
        if sites[i].center == id:
            cnt_sites = cnt_sites+1
            sum_x = sum_x + sites[i].x_location
            sum_y = sum_y + sites[i].y_location
    center.x_location = sum_x/cnt_sites
    center.y_location = sum_y/cnt_sites
    return center

def k_means(sites, init_centers,algorithm_kind):
    centers = init_centers[:]
    n_fig = 0
    while True:
        new_centers = []
        changed_centers = []
        # assign the center to the site
        for site in sites:
            center = assign_center(centers, site)
            site.center = center.id
            center.sites.append(site)
        # recalculate center
        for center in centers:
            new_center = cal_center(center.id, center.sites)
            new_centers.append(new_center)
            # if ((new_center.x_location != center.x_location) or (new_center.y_location != center.y_location)):
            if (((new_center.x_location-center.x_location)>0.01) or ((new_center.y_location-center.y_location)>0.01)):
                changed_centers.append(new_center)

        if len(changed_centers) == 0:
            return centers,n_fig
        centers = new_centers[:]

        plt.clf()
        color_squence = ['darkorchid','limegreen','sandybrown','lightslategrey','rosybrown','sienna','seagreen']
        for j in range(len(centers)):
            x_sample_location = []
            y_sample_location = []
            for i in range(len(sites)):
                if sites[i].center == j:
                    x_sample_location.append(sites[i].x_location)
                    y_sample_location.append(sites[i].y_location)  
            plt.scatter(x_sample_location,y_sample_location,marker='o',c = color_squence[j%7])

        x_center_location = []
        y_center_location = []
        for i in range(len(centers)):
            x_center_location.append(centers[i].x_location)
            y_center_location.append(centers[i].y_location)    
        plt.scatter(x_center_location,y_center_location,s=400,marker='*',c='red')
        plt.title('Init by '+ algorithm_kind)
        plt.xlabel('Number of iterations:' + str(n_fig+1))
        plt.savefig(str(algorithm_kind)+ '_' + str(n_fig)+'.png')
        n_fig = n_fig+1
        plt.pause(0.5)

class Site:
    center = 0
    def __init__(self,id,x_location,y_location):
        self.id = id
        self.x_location = x_location
        self.y_location = y_location

class Center:
    sites = []
    def __init__(self,id,x_location,y_location):
        self.id = id
        self.x_location = x_location
        self.y_location = y_location

if __name__ == "__main__":

    data = pd.read_csv("data.txt",sep='\s',engine='python')
    data = data.values
                                                                                                                                                                                                                                             
    '''
    Define the clusters using super param
    '''   
    K = 2
    N_SAMPLES = len(data) # numbel of samples, K samples is used for initial centers

    '''
    Initial sample sites
    '''
    Object_sites = []
    sample_sites_locations = []
    x_sample_location = data[:,0]
    x_sample_location = x_sample_location.tolist()
    y_sample_location = data[:,1]
    y_sample_location = y_sample_location.tolist()

    for n_sample in range(N_SAMPLES):
        # get sites locations [[x,y],[]...],then initial sites objects
        sample_sites_locations.append([x_sample_location[n_sample],y_sample_location[n_sample]])
        Object_sites.append( Site(n_sample,sample_sites_locations[n_sample][0],sample_sites_locations[n_sample][1]) )

    #############################################################################
    # new method
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    # initial centers randomly
    algorithm_kind = 'Random Initialization '
    rand_squence = np.random.randint(0,N_SAMPLES,K) # choose sites randomly
    for k in range(K):
        # choose sites as centers from samples
        x_center_location.append(x_sample_location[rand_squence[k]])
        y_center_location.append(y_sample_location[rand_squence[k]])
        
        # get centers locations [[x,y],[]...],then initial centers objects
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
        Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    plt.title('Init by '+ algorithm_kind)
    # plt.savefig('inital_centers_Random.png')

    '''
    K-Means and plt.show
    '''
    plt.ion()
    [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)

    center_sitecnt1 = 0
    center_sitecnt2 = 0
    Object_sites1 =[]
    Object_sites2 =[]

    for i in range(N_SAMPLES):
        if Object_sites[i].center == 1:
            Object_sites[i].center = Object_sites[i].center + 3
            center_sitecnt2 = center_sitecnt2 + 1
            Object_sites2.append(Object_sites[i])
        else:
            center_sitecnt1 = center_sitecnt1 + 1
            Object_sites1.append(Object_sites[i])

    for i in range(K):
        if Object_centers[i].id == 1:
            Object_centers[i].id = Object_centers[i].id + 3

    K = 8
    Object_centers.pop()
    Object_centers.pop()

    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    print("objnum:",len(Object_sites1))
    print(center_sitecnt1)
    rand_squence = np.random.randint(0,center_sitecnt1,3) # choose sites randomly
    print(rand_squence)
    if len(Object_sites2) < len(Object_sites1):
        for k in range(3):
            # choose sites as centers from samples
            print(k)
            print(rand_squence[k])
            x_center_location.append(Object_sites1[rand_squence[k]].x_location)
            y_center_location.append(Object_sites1[rand_squence[k]].y_location)
            
            # get centers locations [[x,y],[]...],then initial centers objects
            initial_centers_locations.append([x_center_location[k],y_center_location[k]])
            Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

        initial_centers_locations= []
        x_center_location = []
        y_center_location = []
        rand_squence = np.random.randint(0,center_sitecnt2,5) # choose sites randomly
        for k in range(5):
            # choose sites as centers from samples
            x_center_location.append(Object_sites2[rand_squence[k]].x_location)
            y_center_location.append(Object_sites2[rand_squence[k]].y_location)
            
            # get centers locations [[x,y],[]...],then initial centers objects
            initial_centers_locations.append([x_center_location[k],y_center_location[k]])
            Object_centers.append( Center(k+3,initial_centers_locations[k][0],initial_centers_locations[k][1]) )
    else:
        for k in range(3):
            # choose sites as centers from samples
            print(k)
            print(rand_squence[k])
            x_center_location.append(Object_sites2[rand_squence[k]].x_location)
            y_center_location.append(Object_sites2[rand_squence[k]].y_location)
            
            # get centers locations [[x,y],[]...],then initial centers objects
            initial_centers_locations.append([x_center_location[k],y_center_location[k]])
            Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

        initial_centers_locations= []
        x_center_location = []
        y_center_location = []
        rand_squence = np.random.randint(0,center_sitecnt2,5) # choose sites randomly
        for k in range(5):
            # choose sites as centers from samples
            x_center_location.append(Object_sites1[rand_squence[k]].x_location)
            y_center_location.append(Object_sites1[rand_squence[k]].y_location)
            
            # get centers locations [[x,y],[]...],then initial centers objects
            initial_centers_locations.append([x_center_location[k],y_center_location[k]])
            Object_centers.append( Center(k+3,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)


    plt.ioff()
    plt.show()

    
















    
    
    
    
    
    
    
    # '''
    # Inital centers
    # '''
    # #############################################################################
    # # Random
    # Object_centers = []
    # initial_centers_locations= []
    # x_center_location = []
    # y_center_location = []

    # # initial centers randomly
    # algorithm_kind = 'Random Initialization '
    # rand_squence = np.random.randint(0,N_SAMPLES,K) # choose sites randomly
    # for k in range(K):
    #     # choose sites as centers from samples
    #     x_center_location.append(x_sample_location[rand_squence[k]])
    #     y_center_location.append(y_sample_location[rand_squence[k]])
    #     # pop sites which is chosed as the centers
    #     x_sample_location.pop(rand_squence[k])
    #     y_sample_location.pop(rand_squence[k])
    #     sample_sites_locations.pop(rand_squence[k])
    #     Object_sites.pop(rand_squence[k])
    #     # get centers locations [[x,y],[]...],then initial centers objects
    #     initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    #     Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    # fig = plt.figure(figsize=(5,5))
    # plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    # plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    # plt.title('Init by '+ algorithm_kind)
    # plt.savefig('inital_centers_Random.png')

    # '''
    # K-Means and plt.show
    # '''
    # plt.ion()
    # [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)
    # plt.ioff()
    # plt.show()

    # '''
    # Save figs as gif
    # '''
    # im = Image.open(str(algorithm_kind) + "_0.png")
    # images=[]
    # for i in range(n_fig):
    #     if i!=0:
    #         fpath = str(algorithm_kind) + '_' + str(i) + ".png"
    #         images.append(Image.open(fpath))
    # im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)


    # #############################################################################
    # # density center
    # Object_centers = []
    # initial_centers_locations= []
    # x_center_location = []
    # y_center_location = []

    # # iniital centers using density
    # algorithm_kind = 'DBCIM'
    # [distance_matrix,distance_list] = cal_distance(Object_sites)
    # print(3)
    # radius = 0.5 * max(distance_list) / K
    # density = cal_density(Object_sites,distance_matrix,radius)
    # mindoc = sum(density) / len(density)

    # for i in range(len(density)):
    #     if density[i] < mindoc:
    #         density[i] = 0
    # pre_siteid_of_center = []
    # pre_siteid_of_center.append(density.index(max(density)))
    # x_center_location.append(x_sample_location[pre_siteid_of_center[0]])
    # y_center_location.append(y_sample_location[pre_siteid_of_center[0]])
    # initial_centers_locations.append([x_center_location[0],y_center_location[0]])
    # Object_centers.append( Center(0,x_center_location[0],y_center_location[0]))
    # # plot the initial centers and sites
    # angles_circle = [i * pi / 180 for i in range(0, 360)]  # i change to double
    # x = cos(angles_circle)
    # y = sin(angles_circle)
    # fig = plt.figure(figsize=(5,5))
    # plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    # plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    # plt.plot(x+x_center_location[0],y+y_center_location[0],'r')

    # for i in range(K-1):
    #     pre_siteid_of_center.append(cal_nextcenter(density,distance_matrix,Object_centers,pre_siteid_of_center))
    #     x_center_location.append(x_sample_location[pre_siteid_of_center[i+1]])
    #     y_center_location.append(y_sample_location[pre_siteid_of_center[i+1]])
    #     initial_centers_locations.append([x_center_location[i+1],y_center_location[i+1]])
    #     Object_centers.append( Center(i+1,x_center_location[i+1],y_center_location[i+1]))

    #     # # plot the initial centers and sites
    #     plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    #     plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    #     plt.plot(x+x_center_location[i+1],y+y_center_location[i+1],'r')
          
    # plt.title('Init by '+ algorithm_kind)
    # plt.savefig('inital_centers_DBCIM.png')

    # '''
    # K-Means and plt.show
    # '''
    # plt.ion()
    # [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)
    # plt.ioff()
    # plt.show()

    # # '''
    # # Save figs as gif
    # # '''
    # # im = Image.open(str(algorithm_kind) + "_0.png")
    # # images=[]
    # # for i in range(n_fig):
    # #     if i!=0:
    # #         fpath = str(algorithm_kind) + '_' + str(i) + ".png"
    # #         images.append(Image.open(fpath))
    # # im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)
    
