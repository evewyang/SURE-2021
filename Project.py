import math
from numpy import random
import torch
import numpy as np
import matplotlib.pyplot as plt
device=torch.device("cpu")

#Generate data points from a unit sphere
def random_ball(num_points, dimension, seed, radius=1):
    np.random.seed(seed)
    random_directions = np.random.uniform(low=-1,high=1,size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    return radius * (random_directions * radius).T

#Compute the smallest eigenvalue of H^{\infty}
def get_lambda0(X,N):
    H_infty = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            k = X[i].dot(X[j])/(np.linalg.norm(X[i])*np.linalg.norm(X[j]))
            if k > 1:
                k = 1
            prob = (math.pi - math.acos(k)) / (2*math.pi)
            H_infty[i][j] = X[i].dot(X[j]) * prob
    eigenvalue, featurevector = np.linalg.eig(H_infty)
    return np.amin(eigenvalue)

def generate_list(number_of_sample,length):
    my_list=[]
    count=0
    while count<number_of_sample:
        temp=np.random.randint(0,length)
        if temp not in my_list:
            my_list.append(temp)
            count+=1
    return my_list


#Train the neural network with 2 layers
def train(X,y,D_in,D_out,H,iters,learning_rate,seed,number_of_sample):
    torch.manual_seed(seed)
    w_1 = torch.randn(D_in,H,device=device)
    torch.manual_seed(seed)
    w_2 = -1 + 2 * torch.tensor(np.random.randint(0,2,(H,D_out)),device=device).float()
    #w_2 = torch.tensor(np.random.uniform(-1,1,(H,D_out)),device=device).float()
    losses = []
    size_of_X=X.size()
    n=size_of_X[0]

    for it in range(iters):
        X_copy=X
        y_copy=y
        mini_batch=generate_list(number_of_sample,n)
        h = X_copy[mini_batch].mm(w_1)
        h_relu = h.clamp(min=0)
        y_hat = h_relu.mm(w_2) * 1/math.sqrt(H) 

        loss = (y_hat-y_copy[mini_batch]).pow(2).sum()/2
        losses.append(loss.item())

        y_hat_grad = y_hat - y_copy[mini_batch]
        w_2_grad = h_relu.t().mm(y_hat_grad) * 1/math.sqrt(H)
        h_relu_grad = y_hat_grad.mm(w_2.t())
        h_grad = h_relu_grad.clone()
        h_grad[h<0] = 0
        w_1_grad = X_copy[mini_batch].t().mm(h_grad) * 1/math.sqrt(H)

        w_1 = w_1-learning_rate * w_1_grad
        w_2 = w_2-learning_rate * w_2_grad
    
    return losses  

if __name__ == "__main__":
    #Define the input shape, number of data points
    #Define the number of iterations and learning rate
    N,D_in,D_out=20,100,1
    iters = 2000
    learning_rate = 0.03

    X = random_ball(N, D_in, seed=42)


    lambda0 = get_lambda0(X,N)
    X = torch.tensor(X,device=device).float()
    y = torch.randn(N,D_out,device=device)
    
    #Plot the experimental results
    plt.figure(figsize=(12, 6))
    i = 0
    result = []
    for H in [10,20,100,500,1000,5000]:
        losses = train(X,y,D_in,D_out,H,iters,learning_rate,seed=4396,number_of_sample=10)
        converge = [losses[0] * (1-learning_rate*lambda0/2)**t for t in range(iters)]
        plt.subplot(230+i+1)
        plt.plot(range(iters),np.log(losses),label="LHS")
        plt.plot(range(iters),np.log(converge),label="RHS")
        plt.title("Number of Hidden States: "+str(H))
        plt.xlabel("Epochs")
        plt.ylabel("log(Training Loss)")
        plt.legend(loc="upper right")
        print("Number of Hidden States: "+str(H)+"; " + "Initial Loss:",losses[0])
        result.append([losses,H])
        i += 1
    print("Lambda0:",lambda0)
    print("Learning Rate;",learning_rate)
    print("Linear Convergence Rate:",(1-learning_rate*lambda0/2))
    plt.subplots_adjust(wspace =0.3, hspace =0.5)
    plt.show()

    for term in range(len(result)):
        loss = result[term][0]
        plt.plot(range(iters),np.log(loss),label=str(result[term][1]))
    plt.xlabel("Epochs")
    plt.ylabel("log(Training Loss)")
    plt.legend(loc="lower left")
    plt.title("Learning Rate: "+str(learning_rate))
    plt.show()
