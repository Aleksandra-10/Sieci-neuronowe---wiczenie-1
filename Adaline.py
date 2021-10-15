import random
import numpy as np

#Aleksandra Raszewska
class Adaline:
    def zbior_wartosci(self, ilosc_zmiennych):
        x = []
        d = []
        x.append([-1, -1, 1])
        d.append(-1)
        x.append([-1,1, 1])
        d.append(-1)
        x.append([-1,1, 1])
        d.append(-1)
        x.append([1,1, 1])
        d.append(1)
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
                    d.append(-1)
        return x, d

    def calkowite_pobudzenie(self, x, w):
        suma=0
        for a in range (0, len(x)):
            suma+=x[a]*w[a]
        return suma
    
    def uaktualnienie_wagi(self, x, y, d, wagi, mi):
        for i in range (0, len(x)):
            delta = (d - y) * x[i]
            wagi[i]+= mi*delta
        return wagi, (d - y)**2

    def funkcja_aktywacji_bipolarna(self, suma, prog):
        return np.where(suma>=prog, 1, -1)

    def algorytm_uczenia(self, prog, mi):
        x, d = self.zbior_wartosci(100)
        ile_epok_lista=[]
        for b in range (0, 10):
            wagi  = []
            for i in range(len(d)):
                wagi.append([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])
            liczba_epok = 0
            blad_sredniokwadratowy=500
            while blad_sredniokwadratowy >= prog:
                tablica_epsilon=[]
                liczba_epok+=1
                for a in range(len(d)):
                    suma = self.calkowite_pobudzenie(x[a], wagi[a])
                    #print("x:"+str(x[a]), str(suma))
                    wagi[a], roznica = self.uaktualnienie_wagi(x[a], suma, d[a], wagi[a], mi)
                    tablica_epsilon.append(roznica)
                blad_sredniokwadratowy = sum(tablica_epsilon)/len(tablica_epsilon)
                #print("Błąd średniokwadratowy: "+str(blad_sredniokwadratowy))
            ile_epok_lista.append(liczba_epok)
            suma = self.calkowite_pobudzenie([0.99,0.99], wagi[a]) #test poprawności
            print("Test:"+str(self.funkcja_aktywacji_bipolarna(suma, 0)))
        print("ile epok: "+str(sum(ile_epok_lista)/10))

if __name__ == "__main__":
    adaline = Adaline()
    adaline.algorytm_uczenia(0.1, 0.1)
    