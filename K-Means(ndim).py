import random
import math
import pylab as pl

dims = 3
K = 13 #number of clusters
N_TOTAL = 10000 #Total number of inputs created
T_PERCENT = .80 #percentage of total inputs to be used for training
alpha = 150 #sensitity measure: when the average change in means differs by less than this, we stop
n  = int(N_TOTAL * T_PERCENT) #number of inputs to be used for training
tn = N_TOTAL - n #number of inputs to be used for testing
square = 1000 #Length of one side of the usable square
xleft  = -1 * square #lowest x-value of the usable area
xright =  square #Highest x-value of the usable area
yup    =  square #Highest y-value of the usable area
ydown  = -1 * square #lowest y-value of the usable area
zup    = square
zdown  = -1 * square
zup    = square
zdown  = -1 * square
means   = [] #[MeansOrderedPair] the centroids of the clusters
initial_means = []
inputs  = [] #[InputOrderedPair] the inputs for training
tinputs = [] #[InputOrderedPair] the inputs for testing
categories_breakdown = [] #[[ints],[ints],[ints],[ints]] categories_breakdown[i][j] = the indices of the inputs (from inputs[]) that belong in category i after iteration j
num_in_each_category = [] #[[ints],[ints],[ints],[ints]] num_in_each_category[i][j] = len(categories_breakdown[i]) after iteration j
num_training_in_each_category = [] #[ints] num_training_in_each_category[i] = the number of tninputs that were in cluster i before training
colors = ['red','orange','yellow','green','blue','purple','magenta','cyan','deeppink','lime', 'pink','navy','goldenrod']

class InputOrderedPair:
    def __init__(self, p, cat):
        self.p = p
        self.category = cat
class MeansOrderedPair:
    def __init__(self, p):
        self.p = p
    def recenter(self, p):
        self.p = p
#Initialization Functions
def init_means():
    cen = []
    for i in range(K):
        s = []
        for j in range(dims):
            s.append(random.uniform(-1*square,square))
        means.append(MeansOrderedPair(s))
    for l in range(dims):
        cen.append(0)
    means.append(MeansOrderedPair(cen))
def init_inputs():
    for i in range(n):
        s = []
        for d in range(dims):
            s.append(random.uniform(-1*square,square))
        inputs.append(InputOrderedPair(s,-1))
    for i in range(tn):
        s = []
        s.append(random.uniform(-1*square,square))
        tinputs.append(InputOrderedPair(s,-1))
def init_categories():
    for i in range(K):
        categories_breakdown.append([])
        num_in_each_category.append([])
def init_all(uniform_means):
    if uniform_means:
        means = init_means()
    else:
        means = init_means()
    init_inputs()
    init_categories()
#Calculatory Functions
def vector_norm(v): #Norm of vector v
    s = 0
    for i in range(len(v)):
        s += (v[i])**2
    return (math.sqrt(s))
def distance_from_mean_k(i,k):#the distance that input i (as indiced by inputs[]) is from the center of cluster k
    v = []
    for j in range(dims):
        v.append(inputs[i].p[j] - means[k].p[j])
    return(v)
def mean_distance_from_mean_k(m):
    v = []
    for j in range(dims):
        v.append(means[m].p[j] - init_means[m].p[j])
    return(v)
def categorize(i,test):#determines the closest mean from the given input i (as indiced by inputs[]) and assigns it to that cluster; test is boolean to determine which input (inputs or tinputs) should be uesed
    if test:
        inputs_list = tinputs
    else:
        inputs_list = inputs
    least = 100000000
    cat = -1
    distance = 0
    for k in range(K):
        distance = vector_norm(distance_from_mean_k(i,k))
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
    avg = []
    num  = 1
    for d in range(dims):
        avg.append(0)
        for p in range(len(categories_breakdown[k])):
            avg[d] += inputs[p].p[d]
            num += 1
    for i in range(dims):
            avg[i] /= num
    change_v = []
    for j in range(dims):
        change_v.append(means[k].p[j] - avg[j])
    means[k].recenter(avg)
    return(vector_norm(change_v))
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
def plot_by_size_change(cat):#returns appropriate data to plot iteritic membership figures of each cluster (see plot2())
    x = []
    y = []
    for i in range (len(num_in_each_category[cat])):
        x.append(i)
        y.append(num_in_each_category[cat][i])
    return x,y
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
        print("Cluster {} has center {} & contains {} elements".format(k,means[k].p,len(categories_breakdown[k])))
    for p in range(K):
        print("Cluster {} has recentered by ({})".format(p,mean_distance_from_mean_k(p)))
    print("total iterations: {}".format(its))
def print_beginning_means():#Prints the current (x,y) of each cluster's mean; meant for use at beginning
    for k in range(K):
        print("Cluster {} has center ({})".format(k,means[k].p))
def print_training_data():#Prints membership attributes for the columns based on the training data only
    for k in range(K):
        print("Cluster {} contains {} elements from the training set".format(k,len(categories_breakdown[k])))
    for i in range(K):
        num_training_in_each_category.append(len(categories_breakdown[i]))
    for p in range(K):
        print("Cluster {} contains a net of {} elements from the training set after training".format(p,round(num_training_in_each_category[p] - len(categories_breakdown[p]))))
#Training Functions
def train():#K-Means algorithm; should_plot determines whether a plot should be printed at the start; returns total number of iterations
    categorize_all(False)
    t = alpha + 1
    its = 0
    while t > alpha:
        t = recenter_all_means()
        categorize_all(False)
        track_group_changes()
        print_iterate_data(its,t)
        its += 1
    return(its)
#Run the program
def run(uniform_means):#runs train and prints all data and plots
    init_all(uniform_means)
    categorize_all(True)
    print("Before training: ")
    print_training_data()
    print_beginning_means()
    its= train()
    print_final_data(its)
    categorize_all(True)
    print("After training: ")
    print_training_data()
run(True)
