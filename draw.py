import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


def plot_profit(profit):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total task")
 
    plt.title('50 tasks in 888s')
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0,300])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
    
eval_profit_list = []
# save testint result...
eval_profit_list.append(1)

 
 
plot_profit(eval_profit_list)