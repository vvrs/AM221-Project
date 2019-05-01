import numpy as np 
import matplotlib.pyplot as plt 
import json 
import seaborn as sns 

sns.set()
FILES  = ["results_fgsm.json","results_bim.json","results_pgd.json","results_random.json"]#,"results_bim.json","results_pgd.json"]
def visualizer(files):
    x_fgsm = []
    y_fgsm = []
    y_bar_fgsm = []
    x_bim = []
    y_bim = []
    y_bar_bim = []
    x_pgd = []
    y_pgd = []
    y_bar_pgd = []

    x_random = []
    y_random = []
    y_bar_random = []

    for filename in files:
        data = json.load(open(filename))
        x = []
        y = []
        y_bar = []
        for eps,score in data.items():
            if filename == "results_fgsm.json":
                x_fgsm.append(eps)
                y_fgsm.append(score)
                y_bar_fgsm.append(np.mean(score))
            if filename == "results_bim.json":
                x_bim.append(eps)
                y_bim.append(score)
                y_bar_bim.append(np.mean(score))
                
            if filename == "results_pgd.json":
                x_pgd.append(eps)
                y_pgd.append(score)
                y_bar_pgd.append(np.mean(score))
            if filename == "results_random.json":
                x_random.append(eps)
                y_random.append(score)
                y_bar_random.append(np.mean(score))
                
        if filename == "results_fgsm.json":
            plt.figure()
            plt.boxplot(y_fgsm,patch_artist=True)
            plt.xticks(np.arange(1,12,step=1),[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
            plt.yticks(np.arange(-20,20, step=5))
            plt.title("FSGM")
            plt.savefig("BoxPlot_fgsm.png".format(filename.split('.')[0]))
        if filename == "results_bim.json":
            plt.figure()
            plt.boxplot(y_bim,patch_artist=True)
            plt.xticks(np.arange(1,12,step=1),[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
            plt.yticks(np.arange(-20,20, step=5))
            plt.title("BIM")
            plt.savefig("BoxPlot_bim.png".format(filename.split('.')[0]))
        if filename == "results_pgd.json":
            plt.figure()
            plt.boxplot(y_pgd,patch_artist=True)
            plt.xticks(np.arange(1,12,step=1),[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
            plt.yticks(np.arange(-20,20, step=5))
            plt.title("PGD")
            plt.savefig("BoxPlot_pgd.png".format(filename.split('.')[0]))
        if filename == "results_random.json":
            plt.figure()
            plt.boxplot(y_random,patch_artist=True)
            plt.xticks(np.arange(1,12,step=1),[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
            plt.yticks(np.arange(-20,20, step=5))
            plt.title("Random")
            plt.savefig("BoxPlot_random.png".format(filename.split('.')[0]))
        # plt.xticks(np.arange(0,0.5, step=0.05))
        
    plt.figure()
    plt.title("Accuracies Vs Epsilon".format(filename.split('.')[0]))
    plt.plot(x_fgsm,y_bar_fgsm,"*-", label="FGSM")
    plt.plot(x_bim,y_bar_bim,"*-", label="BIM")
    plt.plot(x_pgd,y_bar_pgd,"*-", label="PGD")
    plt.plot(x_random,y_bar_random,"*-", label="RANDOM")
    # plt.xticks(np.arange(1,12,step=1),[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
    plt.yticks(np.arange(-20,20, step=5))
    plt.legend()
    plt.savefig("AccVsEps.png".format(filename.split('.')[0]))

if __name__ == "__main__":
    # for file_ in FILES:
    visualizer(FILES)
    plt.show()