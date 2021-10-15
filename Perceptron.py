import numpy as np
import random
from numpy.lib.function_base import percentile

#Aleksandra Raszewska
class Perceptron:
    def zbior_wartosci(self, ilosc_zmiennych, bias, bipolarna):
        a=0
        if bipolarna:
            a=-1
        if bias == False:
            x = []
            d = []
            x.append([a,a])
            d.append(a)
            x.append([a,1])
            d.append(a)
            x.append([1,a])
            d.append(a)
            x.append([1,1])
            d.append(1)
            if a == 0:
                for _ in range(0, ilosc_zmiennych):
                    zakres = random.choice([0,1])
                    if zakres == 0:
                        x1 = random.uniform(0.9, 1.1)
                        x2 = random.uniform(0.9, 1.1)
                        x.append([x1, x2])
                        d.append(1)
                    else:
                        x1 = random.uniform(-0.1, 0.1)
                        x2 = random.uniform(-0.1, 0.1)
                        x.append([x1, x2])
                        d.append(a)
            else:
                for _ in range(0, ilosc_zmiennych):
                    zakres = random.choice([0,1])
                    if zakres == 0:
                        x1 = random.uniform(0.9, 1.1)
                        x2 = random.uniform(0.9, 1.1)
                        x.append([x1, x2])
                        d.append(1)
                    else:
                        x1 = random.uniform(-1.1, 0.9)
                        x2 = random.uniform(-1.1, 0.9)
                        x.append([x1, x2])
                        d.append(a)
        else:
            x = []
            d = []
            x.append([a, a, 1])
            d.append(a)
            x.append([a,1, 1])
            d.append(a)
            x.append([1,a, 1])
            d.append(a)
            x.append([1,1, 1])
            d.append(1)
            if a == 0:
                for _ in range(0, ilosc_zmiennych):
                    zakres = random.choice([0,1])
                    if zakres == 0:
                        x1 = random.uniform(0.9, 1.1)
                        x2 = random.uniform(0.9, 1.1)
                        x.append([x1, x2, 1])
                        d.append(1)
                    else:
                        x1 = random.uniform(-0.1, 0.1)
                        x2 = random.uniform(-0.1, 0.1)
                        x.append([x1, x2, 1])
                        d.append(a)
            else:
                for _ in range(0, ilosc_zmiennych):
                    zakres = random.choice([0,1])
                    if zakres == 0:
                        x1 = random.uniform(0.9, 1.1)
                        x2 = random.uniform(0.9, 1.1)
                        x.append([x1, x2, 1])
                        d.append(1)
                    else:
                        x1 = random.uniform(-1.1, -0.9)
                        x2 = random.uniform(-1.1, -0.9)
                        x.append([x1, x2, 1])
                        d.append(a)

        
        return x, d

    def calkowite_pobudzenie(self, x, w):
        suma=0
        for a in range (0, len(x)):
            suma+=x[a]*w[a]
        return suma

    def funkcja_aktywacji_unipolarna(self, suma, prog):
        return np.where(suma>=prog, 1, 0)

    def funkcja_aktywacji_bipolarna(self, suma, prog):
        return np.where(suma>=prog, 1, -1)

    def uaktualnienie_wagi(self, x, y, d, wagi, alfa):
        a = 0
        for i in range (0, len(x)):
            delta = (d - y) * x[i]
            #print(delta)
            a+=delta
            wagi[i]+= alfa*delta
        return wagi, a

    def algorytm_uczenia(self, alfa, prog, bias, bipolarna):
        x, d = self.zbior_wartosci(100, bias, bipolarna)
        ile_epok_lista=[]
        for b in range (0, 10):
            wagi  = []
            for i in range(len(d)):
                if bias == False:
                    wagi.append([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                else:
                    wagi.append([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
            czy_bez_zmian= -1
            liczba_epok = 0
            while czy_bez_zmian != 0:
                liczba_epok+=1
                czy_bez_zmian = 0
                for a in range(len(d)):
                    suma = self.calkowite_pobudzenie(x[a], wagi[a])
                    wagi[a], delta = self.uaktualnienie_wagi(x[a], self.funkcja_aktywacji_unipolarna(suma, prog),d[a], wagi[a], alfa)
                    if delta != 0:
                        czy_bez_zmian+=delta
                #print("______________________________________")
            ile_epok_lista.append(liczba_epok)
            suma = self.calkowite_pobudzenie([0,0.99], [0.3, 0.5])
            print("Test:"+str(self.funkcja_aktywacji_unipolarna(suma, 0.5)))
        print("ile epok: "+str(sum(ile_epok_lista)/10))


if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.algorytm_uczenia(0.1, 0.1, True, False)
