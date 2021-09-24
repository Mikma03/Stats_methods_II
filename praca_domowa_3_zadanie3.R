
###################### ZADANIE 3 ######################

zakres1<-c(4.22,4.77,3.39,3.78,3.29,2.62,3.40,4.75)

zakres2<-c(3.32,4.46,2.41,1.30,3.33,1.06,2.32,3.17)

zakres3<-c(4.94,6.52,3.80,4.95,5.25,3.67,3.76,5.03)

zmienne<-data.frame(zakres1,zakres2,zakres3)


###################### W dalszej kolejnoœci zostanie przeprowadzony test permutacyjny ######################

###################### Nale¿y porównac okresy parami aby sprawdziæ czy dochody ró¿ni¹ siê od siebie ######################


###################### Policzmy na ile sposobów mo¿na zestawiæ ze sob¹ grupy ######################

factorial(16)/(factorial(8)*factorial(8))/2

###################### Nastêpnie sprawdzimy wszystkie mo¿liwe kombinacje ######################

N=16

x=1:N

x1=combn(x,N/2)

NC=NCOL(x1)

x2=x1[,NC:1]

grupa1=t(x1[,1:(NC/2)])

grupa2=t(x2[,1:(NC/2)])

all.comb=cbind(grupa1,grupa2)

all.comb


###################### zakres1 vs zakres2 ######################

roznice_zakres12<-c()

zakres_12<-c(zakres1,zakres2)

###################### Nastêpnie policzymy œredni¹ dla ka¿dej kombinacji ######################

roznice<-c()

for (i in 1:nrow(all.comb)){
  
  wartosc_sredniej_1<-mean(zakres_12[all.comb[i,1:8]])
  
  wartosc_sredniej_2<-mean(zakres_12[all.comb[i,9:16]])
  
  dif<-abs(wartosc_sredniej_1-wartosc_sredniej_2)
  
  roznice_zakres12<-c(roznice_zakres12,dif)
  
}

###################### sortujemy rosn¹co ######################

roznice_zakres12<-sort(roznice_zakres12) 

roznica12_abs<-abs(mean(zakres1)-mean(zakres2))

###################### W dalszej kolejnoœci sprawdzimy wartoœc alfa/2 oraz 1 - alfa/2 ######################

###################### zak³adamy alfa = 5% ######################

alfa=0.05

kwantyl12_dol<-quantile(roznice_zakres12,alfa/2,names=FALSE)

kwantyl12_gora<-quantile(roznice_zakres12,1-alfa/2,names=FALSE)

if (roznica12_abs<kwantyl12_dol | roznica12_abs>kwantyl12_gora) {
  
  print('Roznica miedzy zakresem 1 i 2 jest istotna')
  
} else {print('Roznica miedzy zakresem 1 i 2 nie jest istotna')}





####################### zakres1 vs zakres3 ######################

###################### metoda postêpowania jest podobna do tej powy¿ej ######################

roznice_zakres13<-c()

zakres_13<-c(zakres1,zakres3)

roznice<-c()

for (i in 1:nrow(all.comb)){
  
  wartosc_sredniej_1<-mean(zakres_13[all.comb[i,1:8]])
  
  wartosc_sredniej_3<-mean(zakres_13[all.comb[i,9:16]])
  
  dif<-abs(wartosc_sredniej_1-wartosc_sredniej_3)
  
  roznice_zakres13<-c(roznice_zakres13,dif)
  
}

roznice_zakres13<-sort(roznice_zakres13)

roznica13_abs<-abs(mean(zakres1)-mean(zakres3))

alfa=0.05

kwantyl13_dol<-quantile(roznice_zakres13,alfa/2,names=FALSE)

kwantyl13_gora<-quantile(roznice_zakres13,1-alfa/2,names=FALSE)

if (roznica13_abs<kwantyl13_dol | roznica13_abs>kwantyl13_gora) {
  
  print('Roznica miedzy zakresem1 i 3 jest istotna')
  
} else {print('Roznica miedzy zakresem1 i 3 nie jest istotna')}



###################### zakres2 vs okreszakres3 ######################

roznice_zakres23<-c()

zakres_23<-c(zakres2,zakres3)

roznice<-c()

for (i in 1:nrow(all.comb)){
  
  wartosc_sredniej_2<-mean(zakres_23[all.comb[i,1:8]])
  
  wartosc_sredniej_3<-mean(zakres_23[all.comb[i,9:16]])
  
  dif<-abs(wartosc_sredniej_2-wartosc_sredniej_3)
  
  roznice_zakres23<-c(roznice_zakres23,dif)
  
}

roznice_zakres23<-sort(roznice_zakres23)

roznica23_abs<-abs(mean(zakres2)-mean(zakres3))

alfa=0.05

kwantyl23_dol<-quantile(roznice_zakres23,alfa/2,names=FALSE)

kwantyl23_gora<-quantile(roznice_zakres23,1-alfa/2,names=FALSE)

if (roznica23_abs<kwantyl23_dol | roznica23_abs>kwantyl23_gora) {
  
  print('Roznica miedzy zakresem 2 i 3 jest istotna')
  
} else {print('Roznica miedzy zakresem 2 i 3 nie jest istotna')}


###################### OdpowiedŸ: pomiêdzy zakresem 2 oraz 3 jest istotna ró¿nica, tylko! ######################

