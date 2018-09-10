import mnist
import numpy as np

### Base de donnée des caractères  ###

images = mnist.train_images()
X_exemple = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))/255 # Liste des matrices des chiffres
Y_desiree = mnist.train_labels()                                                     # Liste des chiffres

### Fonction d'activation : la fonction sigmoïde ###

def f(x,deriver=False):

    if(deriver==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

## Classe du réseau de neurones ###

class Network():

    ### On initialise aléatoirement les coefficients synaptiques ###

    def __init__(self,nSynapses,nNeurones,nEntrees,nSorties):

        self.synapses = [] # Liste des matrices synaptiques qui contiennent les coefficients synaptiques ωij
        self.nSynapses = nSynapses
        self.synapses.append(2*np.random.random((nEntrees,nNeurones)) - 1)      # Première synapse
        for k in range(nSynapses-2):                                            # Synapses des couches cachées ---
            self.synapses.append(2*np.random.random((nNeurones,nNeurones)) - 1) # ---
        self.synapses.append(2*np.random.random((nNeurones,nSorties)) - 1)      # Dernière synapse

    def train(self,X,Y,f): # On entraine le réseau à répondre une sortie Y pour une entrée X d'après une fonction d'activation f
        
        biais = 0.01
        
        # Génération des couches par l'entrée X
        
        Layers = [X] # Liste des couches, la première couche est l'entrée
        for i in range(self.nSynapses):
            Layers.append(f(np.dot(Layers[i],self.synapses[i]))+biais)

        # Erreurs et delta

        LayersError = [] 
        LayersDelta = []
        
        LayersError.append(Y - Layers[-1])
        LayersDelta.append(LayersError[-1]*f(Layers[-1]+biais,deriver=True))
        
        for i in range(len(Layers)-2):
            
            LayersError.append(LayersDelta[-1].dot(self.synapses[-(i+1)].T))
            LayersDelta.append(LayersError[-1]*f(Layers[-(i+2)]+biais,deriver=True))

        # Modification des poids synaptiques

        for i in range(len(self.synapses)):            
            self.synapses[-(i+1)] += np.dot(np.transpose(Layers[-(i+2)]),LayersDelta[i])

    def use(self,X): # On propage une entrée X dans le réseau pour obtenir le vecteur de sortie Y

        biais = 0.01
        Layers = [X]
        for i in range(self.nSynapses):
            Layers.append(f(np.dot(Layers[i],self.synapses[i]))+0.01)
        return Layers[-1]

    def digitToVector(self,index):  # On transforme un chiffre en vecteur de sortie pour entrainer le réseau
        chiffre = Y_desiree[index]
        y = [[0,0,0,0,0,0,0,0,0,0]]
        y[0][chiffre] = 1
        return y

    def trainNetwork(self,nSamples,f):  # On entraine le réseau avec 'nSample' exemples

        for k in range(nSamples):
            x = [X_exemple[k]]
            y = self.digitToVector(k)
            self.train(x,y,f)

    def vectorToDigit(self,vector):  # On transforme un vecteur de sortie en chiffre pour vérifier l'éfficacité du réseau
        maxValue = 0.0
        maxDigit = 0
        for i in range(10):
            if vector[0][i] > maxValue:
                maxValue = vector[0][i]
                maxDigit = i
        return maxDigit

    def getSuccessRate(self,nSample): # On vérifie l'éfficacité du réseau sur 'nSample" exemples

        chiffres_reconnus = 0
        for k in range(nSample):
            vector = self.use([X_exemple[59000 + k]])
            realDigit = Y_desiree[59000 + k]
            if realDigit == self.vectorToDigit(vector):
                chiffres_reconnus += 1

        return chiffres_reconnus/nSample

### Exemple pour créer un réseau de neurone ###

# network = Network(3,40,784,10) # On crée un réseau de neurones de 3 synapses et 40 neurones par couches cachées
# network.trainNetwork(59000,f) # On l'entraine sur 59000 exemples
# rate = network.getSuccessRate(n))) # On vérifie son taux de reconaissance

### Fonctions annexes pour tester les taux de reconnaissance sur plusieurs réseaux à architecture différentes ###

def getAverageRate(nSynapsesMax,nNeuronesMax,n):

    rateList = []
    for nSynapses in range(2,nSynapsesMax+1):
        Lsynapses = []
        for nNeurones in range(10,nNeuronesMax+1,10):
            rateValue = 0
            for k in range(n):
                print(k, nSynapses, nNeurones)
                network = Network(nSynapses, nNeurones, 784, 10)
                network.trainNetwork(59000,f)
                rateValue += network.getSuccessRate(100)
            Lsynapses.append(rateValue)
        rateList.append(Lsynapses)

    return np.array(rateList)/n

def getBestNeuronNumber(nNeuronesMax,n):

    rateList = []
    for nNeurones in range(20,nNeuronesMax+1):
        rateValue = 0
        for k in range(n):
            rateValue = 0
            network = Network(3,nNeurones,748,10)
            network.trainNetwork(59000,f)
            rateValue += network.getSuccessRate(100)
        rateList.append(rateValue)

    return np.array(rateList)/n

### Pour obtenir les résultats finaux ###

#print(getAverageRate(9,50,5))
print(getBestNeuronNumber(50,5))