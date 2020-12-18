import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import metrics

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
    global n_fig
    centers = init_centers[:]
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
            if (((new_center.x_location-center.x_location)>100) or ((new_center.y_location-center.y_location)>100)):
                changed_centers.append(new_center)

        if len(changed_centers) == 0:
            return centers,n_fig
        centers = new_centers[:]

        plt.clf()
        color_squence = ['darkorchid','limegreen','sandybrown','lightslategrey','rosybrown','sienna','seagreen','red','black']
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
        plt.title(algorithm_kind)
        plt.xlabel('Number of iterations:' + str(n_fig+1))
        plt.savefig(str(algorithm_kind)+ '_find_K_' + str(n_fig)+'.png')
        n_fig = n_fig+1
        plt.pause(0.005)

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

    '''
    Read the data
    '''   
    data = pd.read_csv("data.txt",sep='\s',engine='python')
    data = data.values

    '''
    Init sites and centers
    '''
    Object_sites = []
    sample_sites_locations = []
    x_sample_location = data[:,0]
    x_sample_location = x_sample_location.tolist()
    y_sample_location = data[:,1]
    y_sample_location = y_sample_location.tolist()
    N_SAMPLES = len(data) 

    for n_sample in range(N_SAMPLES):
        # get sites locations [[x,y],[]...],then initial sites objects
        sample_sites_locations.append([x_sample_location[n_sample],y_sample_location[n_sample]])
        Object_sites.append( Site(n_sample,sample_sites_locations[n_sample][0],sample_sites_locations[n_sample][1]) )

    '''
    Evaluate K
    '''
    scores1 = []
    scores2 = []
    K_sque = []
    for k in range(8):
        K = k+2
        K_sque.append(K)
        Object_centers = []
        initial_centers_locations= []
        x_center_location = []
        y_center_location = []

        # initial centers randomly
        np.random.seed(10)
        rand_squence = np.random.randint(0,N_SAMPLES,K) # choose sites randomly
        for k in range(K):
            # choose sites as centers from data
            x_center_location.append(x_sample_location[rand_squence[k]])
            y_center_location.append(y_sample_location[rand_squence[k]])
            
            # get centers locations [[x,y],[]...],then initial centers objects
            initial_centers_locations.append([x_center_location[k],y_center_location[k]])
            Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

        algorithm_kind = 'K-Means when K = ' + str(K)
        '''
        Use K-Means for the first time 
        '''
        plt.ion()
        # use K-Means for the first time
        global n_fig
        n_fig = 0
        [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)

        plt.ioff()
        plt.show()

        index = []
        for i in range(len(Object_sites)):
            index.append(Object_sites[i].center+1)
        score1 = metrics.calinski_harabasz_score(data, index)
        score2 = metrics.silhouette_score(data,index,metric='euclidean')
        scores1.append(score1)
        scores2.append(score2)
    plt.plot(K_sque,scores1,'*-')
    plt.title("calinski_harabasz_score")
    plt.xlabel("K")
    plt.show()

    plt.plot(K_sque,scores2,'o-')
    plt.title("silhouette_score")
    plt.xlabel("K")
    plt.show()

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind) + "_find_K_0.png")
    images=[]
    for i in range(n_fig):
        if i!=0:
            fpath = str(algorithm_kind) + '_find_K_' + str(i) + ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)

