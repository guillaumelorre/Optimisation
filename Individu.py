#coding: -utf8


import numpy as np
import random
import math



class Individu:



    def __init__(self, n, x0): # n = dimension du problème
        self.n = n
        self.x = x0
        #self.x = np.random.randn(n)  # A revoir plus tard l'initialisation
        self.delta = np.ones(n)#np.random.randn(n) +1
        self.deltaR=2.0
        #self.deltaR = random.gauss(0,1)
        self.r = np.random.randn(n)
        self.s = np.zeros(n)
        self.sr = 2.0


    def mutation(self):

        # Définition des paramètres

        c = 1.0/math.sqrt(self.n)
        beta = 1.0/self.n
        xn = math.sqrt(self.n)*(1.0-1.0/(4*self.n) + 1.0/(21*self.n*self.n))
        x1 = math.sqrt(2.0/math.pi)
        bInd = 1.0/(4*self.n)
        cr = 3.0/(self.n+3)
        br = 1.0/math.sqrt(4*self.n)

        #Mutation of the object variables

        z = np.random.randn(self.n)
        zR = random.gauss(0,1)
        enfant = Individu(self.n, self.x + self.delta*z + self.deltaR*zR*self.r)
        enfant.x = self.x + self.delta*z + self.deltaR*zR*self.r

        #Adaptation of the individual step size

        cu = math.sqrt((2-c)/c)
        enfant.s = (1-c)*self.s + c*cu*z

        #enfant.delta = self.delta*math.exp(beta*np.linalg.norm(enfant.s)-xn)*np.array([math.exp(bInd*(abs(x)-x1)) for x in enfant.s])
        enfant.delta = self.delta*np.array([math.exp(bInd*(abs(x)-x1)) for x in enfant.s])
        #print("param delta : ", math.exp(beta*np.linalg.norm(enfant.s)-xn))

        # Direction adaptation

        enfant.sr = max([0, (1.0-c)*self.sr + c*cu*zR])
        rPrime = (1.0-cr)*self.deltaR*self.r + cr*(enfant.x-self.x)
        enfant.r = rPrime/np.linalg.norm(rPrime)
        enfant.deltaR = max([self.deltaR*math.exp(br*(enfant.sr-x1)),1.0/3*np.linalg.norm(enfant.delta)])


        return enfant

class Population :
    def __init__(self, fonction, n,  mu, lan, x0):
        self.f= fonction
        self.n = n
        self.lan=lan
        self.mu=mu
        self.individus = []
        for i in range(mu) :
            self.individus.append(Individu(n, x0))
    def nextGen(self):
        enfants=[]
        for i in range(self.lan):
            xi = random.randint(0, self.mu-1)
            e = self.individus[xi].mutation()
            enfants.append([e, self.f(e.x)])
        
        enfants = sorted(enfants, key=lambda colonnes : colonnes[1])
        self.individus=[]
        for i in range(self.mu) :
            self.individus.append(enfants[i][0])


    
def rosenbrock(chromosome):
	"""F8 Rosenbrock's saddle
	multimodal, asymmetric, inseparable"""
	fitness = 0
	for i in range(len(chromosome)-1):
		fitness += 100*((chromosome[i]**2)-chromosome[i+1])**2+(1-chromosome[i])**2
	return fitness

       
def f(x) : 
    return np.dot(x,x)
        
def A2(fun, x0, budget):
    pop = Population(fun, len(x0) , 1, 10, x0)
    for i in range(budget):
        pop.nextGen()
        #print func(pop.individus[0].x)
    return pop.individus[0].x
    
    
        
#A2(rosenbrock, np.zeros(10), 1000000)  
        




