import random
import math
import pylab as pl

K = 13 #number of clusters
N_TOTAL = 10000 #Total number of inputs created
T_PERCENT = .80 #percentage of total inputs to be used for training
alpha = .001 #sensitity measure: when the average change in means differs by less than this, we stop
n  = int(N_TOTAL * T_PERCENT) #number of inputs to be used for training
tn = N_TOTAL - n #number of inputs to be used for testing
square = 1000 #Length of one side of the usable square
xleft  = -1 * square #lowest x-value of the usable area
xright =  square #Highest x-value of the usable area
yup    =  square #Highest y-value of the usable area
ydown  = -1 * square #lowest y-value of the usable area
zup    = square
zdown  = -1 * square
means   = [] #[MeansOrderedPair2D] the centroids of the clusters
initial_means = []
inputs  = [] #[InputOrderedPair2D] the inputs for training
tinputs = [] #[InputOrderedPair2D] the inputs for testing
categories_breakdown = [] #[[ints],[ints],[ints],[ints]] categories_breakdown[i][j] = the indices of the inputs (from inputs[]) that belong in category i after iteration j
num_in_each_category = [] #[[ints],[ints],[ints],[ints]] num_in_each_category[i][j] = len(categories_breakdown[i]) after iteration j
num_training_in_each_category = [] #[ints] num_training_in_each_category[i] = the number of tninputs that were in cluster i before training
colors = ['red','orange','yellow','green','blue','purple','magenta','cyan','deeppink','lime', 'pink','navy','goldenrod']

class InputOrderedPair2D:
    def __init__(self, x, y, cat):
        self.x_coordinate = x
        self.y_coordinate = y
        self.category = cat
class MeansOrderedPair2D:
    def __init__(self, x, y):
        self.x_coordinate = x
        self.y_coordinate = y
    def recenter(self,x,y):
        self.x_coordinate = x
        self.y_coordinate = y
#Initialization Functions
def square_means_2D(i):
    s0 = MeansOrderedPair2D(.5*xright,.5*yup)
    s1 = MeansOrderedPair2D(.5*xleft,.5*yup)
    s2 = MeansOrderedPair2D(.5*xleft,.5*ydown)
    s3 = MeansOrderedPair2D(.5*xright,.5*ydown)
    smeans = [s0,s1,s2,s3]
    return smeans[i]
def in_between_means_2D(i,j):
    imeanx = (means[i].x_coordinate + means[j].x_coordinate) / 2
    imeany = (means[i].y_coordinate + means[j].y_coordinate) / 2
    ib = MeansOrderedPair2D(imeanx,imeany)
    return ib
def center_mean_2_2D():
    c = MeansOrderedPair2D(0,0)
    return c
def expand_corner(i):
    if i == 0:
        point = in_between_means_2D(4,7)
    else:
        point = in_between_means_2D(i + 3, i + 4)
    xexp = means[i].x_coordinate + point.x_coordinate
    yexp = means[i].y_coordinate + point.y_coordinate
    corner = MeansOrderedPair2D(xexp,yexp)
    return(corner)
def pinch_corners(i):
    if i == 0:
        point = in_between_means_2D(4,7)
    else:
        point = in_between_means_2D(i + 3, i + 4)
    return(point)
def init_means():
    if K < 4:
        if K == 1 or K == 3:
            means.append(center_mean_2_2D())
        if K > 1:
            means.append(square_means_2D(0))
            means.append(square_means_2D(2))
    elif K > 3:
        for i in range(4):
            means.append(square_means_2D(i))
    if K > 7:
        for j in range(K - 4):
            if j > 2:
                break
            means.append(in_between_means_2D(j,j+1))
        means.append(in_between_means_2D(3,0))
    else:
        if K > 5:
            means.append(in_between_means_2D(0,1))
            means.append(in_between_means_2D(2,3))
        if K == 5 or K == 7:
            means.append(center_mean_2_2D())
    if K > 8:
        means.append(center_mean_2_2D())
        for p in range(4):
            if p > K - 10:
                break
            cm = expand_corner(p)
            means.append(cm)
            means[p] = pinch_corners(p)
    for i in range(K):
        m = MeansOrderedPair2D(means[i].x_coordinate,means[i].y_coordinate)
        initial_means.append([])
        initial_means[i] = m
    return means
def init_means_random():
    for i in range(K):
        means.append(MeansOrderedPair2D(random.uniform(xleft,xright), random.uniform(ydown,yup)))
    for j in range(K):
        m = MeansOrderedPair2D(means[i].x_coordinate,means[i].y_coordinate)
        initial_means.append([])
        initial_means[i] = m
def init_inputs():
    for i in range(n):
        inputs.append(InputOrderedPair2D(random.uniform(xleft,xright), random.uniform(ydown,yup), -1))
    for i in range(tn):
        tinputs.append(InputOrderedPair2D(random.uniform(xleft,xright), random.uniform(ydown,yup), -1))
def init_categories():
    for i in range(K):
        categories_breakdown.append([])
        num_in_each_category.append([])
def init_all(uniform_means):
    if uniform_means:
        means = init_means()
    else:
        means = init_means_random()
    init_inputs()
    init_categories()
#Calculatory Functions
def vector_norm(v): #Norm of vector v
    s = 0
    for i in range(len(v)):
        s += (v[i])**2
    return (math.sqrt(s))
def distance_from_mean_k(i,k):#the distance that input i (as indiced by inputs[]) is from the center of cluster k
    v = [inputs[i].x_coordinate - means[k].x_coordinate, inputs[i].y_coordinate - means[k].y_coordinate]
    return vector_norm(v)
def categorize(i,test):#determines the closest mean from the given input i (as indiced by inputs[]) and assigns it to that cluster; test is boolean to determine which input (inputs or tinputs) should be uesed
    if test:
        inputs_list = tinputs
    else:
        inputs_list = inputs
    least = 100000000
    cat = -1
    distance = 0
    for k in range(K):
        distance = distance_from_mean_k(i,k)
        if distance < least:
            least = distance
            cat = k
    inputs_list[i].category = cat
    categories_breakdown[cat].append(i)
def categorize_all(test):#iterative function for all inputs, calls categorize for each
    for k in range(K):
        categories_breakdown[k] = []
    if test:
        for i in range(tn):
            categorize(i,True)
    else:
        for i in range(n):
            categorize(i,False)
def recenter_mean(k):#calculates the average x and average y value for all data points in a given cluster and sets that as the cluser's new mean
    avgx = 0
    avgy = 0
    num  = 1
    for i in categories_breakdown[k]:
        avgx += inputs[i].x_coordinate
        avgy += inputs[i].y_coordinate
        num  += 1
    avgx /= num
    avgy /= num
    change_v = [means[k].x_coordinate - avgx, means[k].y_coordinate - avgy]
    means[k].recenter(avgx,avgy)
    return vector_norm(change_v)
def recenter_all_means():#iterative for recenter_mean_2
    t = 0
    for k in range(K):
        t += recenter_mean(k)
    return t
#Graphical Display Functions
def track_group_changes():#adds to num_in_each_category to keep iteritic data on group membership figures
    for i in range(K):
        num_in_each_category[i].append(len(categories_breakdown[i]))
def least(i):#return smallest num_in_each_category[i] value; for graph
    least = n
    for l in num_in_each_category[i]:
        if l < least:
            least = l
    return least
def most(i):#return largest num_in_each_category[i] value; for graph
    most = 0
    for l in num_in_each_category[i]:
        if l > most:
            most = l
    return most
def total_least():#iterative for least(i); for graph
    l = n
    for i in range(K):
        if least(i) < l:
            l = least(i)
    return l
def total_most():#iterative for most(i); for graph
    m = 0
    for i in range(K):
        if most(i) > m:
            m = most(i)
    return m
def plot_by_category(cat):#returns appropriate data to plot membership of each category (see plot())
    x = []
    y = []
    for i in categories_breakdown[cat]:
        x.append(inputs[i].x_coordinate)
        y.append(inputs[i].y_coordinate)
    return x,y
def plot_by_size_change(cat):#returns appropriate data to plot iteritic membership figures of each cluster (see plot2())
    x = []
    y = []
    for i in range (len(num_in_each_category[cat])):
        x.append(i)
        y.append(num_in_each_category[cat][i])
    return x,y
def plot_means_movements_x(cat):
    return [initial_means[cat].x_coordinate,means[cat].x_coordinate]
def plot_means_movements_y(cat):
    return [initial_means[cat].y_coordinate,means[cat].y_coordinate]
def plot(title):#Plots the dotted graph of each cluster
    x = []
    y = []
    cx = []
    cy = []
    mx = []
    my = []
    for i in range(K):
        cx.append([])
        cy.append([])
    for j in range(len(means)):
        cx[j],cy[j] = plot_by_category(j)
    for k in range(K):
        x.append(means[k].x_coordinate)
        y.append(means[k].y_coordinate)
    for p in range(K):
        mx.append([])
        my.append([])
        mx[p] = initial_means[p].x_coordinate
        my[p] = initial_means[p].y_coordinate
    pl.xlim(xleft - 0.25*square, xright + .25*square)
    pl.ylim(ydown - .25*square, yup + .25*square)
    for l in range(K):
        pl.plot([0,0],[10,10],'-k')
        pl.plot(plot_means_movements_x(l),plot_means_movements_y(l), '-k')
        pl.scatter(cx[l],cy[l],color=colors[l],label="Group {}".format(l))
    pl.scatter(x,y,color='black',label='Means')
    pl.plot(mx,my,'k^',label='Original Means')
    pl.legend(loc='upper right')
    pl.title(title)
    pl.show()
def plot2(its):#Plots the line graphs of total group membersihp by iteration
    ax = []
    ay = []
    for i in range(K):
        ax.append([])
        ay.append([])
    for j in range(K):
        ax[j],ay[j] = plot_by_size_change(j)
    pl.figure(2)
    pl.xlim(0,its-1)
    pl.ylim(total_least() - 5,total_most() + 5)
    for k in range(K):
        pl.plot(ax[k],ay[k],color=colors[k],label='Group {}'.format(k))
    pl.legend(loc='upper right')
    pl.title('Cluster Membership Totals by Iteration')
    pl.xlabel('Iterations')
    pl.ylabel('Total Number in Cluster')
    pl.show()
def print_iterate_data(its,t):#Prints data to be recorded every interation
    print("Average change in means @ iteration {} = {}".format(its,t))
def print_final_data(its):#Prints data needed after the program has stopped
    for i in range(K):
        print("Group {} ranged in size from {} to {}".format(i,least(i),most(i)))
    for j in range(K):
        print("Group {} started at size {} and finished at size {}, for a net gain of {}".format(j,num_in_each_category[j][0],num_in_each_category[j][its - 1], num_in_each_category[j][its-1] - num_in_each_category[j][0]))
    for l in range(K):
        print("Group {}'s sizes: {}".format(l, num_in_each_category[l]))
    for k in range(K):
        print("Cluster {} has center ({},{}) & contains {} elements".format(k,round(means[k].x_coordinate,2),round(means[k].y_coordinate,2),len(categories_breakdown[k])))
    for p in range(K):
        print("Cluster {} has recentered by ({},{})".format(p,round(means[p].x_coordinate - initial_means[p].x_coordinate,2),round(means[p].y_coordinate - initial_means[p].y_coordinate,2)))
    print("total iterations: {}".format(its))
def print_beginning_means():#Prints the current (x,y) of each cluster's mean; meant for use at beginning
    for k in range(K):
        print("Cluster {} has center ({},{})".format(k,means[k].x_coordinate,means[k].y_coordinate))
def print_training_data():#Prints membership attributes for the columns based on the training data only
    for k in range(K):
        print("Cluster {} contains {} elements from the training set".format(k,len(categories_breakdown[k])))
    for i in range(K):
        num_training_in_each_category.append(len(categories_breakdown[i]))
    for p in range(K):
        print("Cluster {} contains a net of {} elements from the training set after training".format(p,round(num_training_in_each_category[p] - len(categories_breakdown[p]))))
#Training Functions
def train(should_plot):#K-Means algorithm; should_plot determines whether a plot should be printed at the start; returns total number of iterations
    categorize_all(False)
    if should_plot:
        plot("Initial Categorization of the Inputs; n = {}".format(n))
    t = 10.
    its = 0
    while t > alpha:
        t = recenter_all_means()
        categorize_all(False)
        track_group_changes()
        print_iterate_data(its,t)
        its += 1
    return(its)
#Run the program
def run(should_plot,uniform_means):#runs train and prints all data and plots
    init_all(uniform_means)
    categorize_all(True)
    print("Before training: ")
    print_training_data()
    if should_plot:
        plot("Initial Distribution of the Testing Data, Before Training; n = {}".format(tn))
    print_beginning_means()
    its= train(should_plot)
    print_final_data(its)
    if should_plot:
        plot2(its)
        plot("Final Distribution of the Training Points with Means Stable; n = {}".format(n))
    categorize_all(True)
    print("After training: ")
    print_training_data()
    if should_plot:
        plot("Distribution of the Testing Points After Training; n = {}".format(tn))
run(True,True)
