import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

new_file = csv.reader(open('input/dataset_original.csv', 'r'),delimiter='_')
cont_h = np.zeros(24)
cont_d = np.zeros(35)
alltext = ""
duracao = []
for row in tqdm(list(new_file)):
    duracao.append(float(row[4]))
    dia = row[1]
    hora = row[2][0:2]
    if "20200101" == dia:
        cont_d[0] += 1
    if "20200102" == dia:
        cont_d[1] += 1
    if "20200103" == dia:
        cont_d[2] += 1
    if "20200104" == dia:
        cont_d[3] += 1
    if "20200105" == dia:
        cont_d[4] += 1
    if "20200106" == dia:
        cont_d[5] += 1
    if "20200107" == dia:
        cont_d[6] += 1
    if "20200108" == dia:
        cont_d[7] += 1
    if "20200109" == dia:
        cont_d[8] += 1
    if "20200110" == dia:
        cont_d[9] += 1
    if "20200111" == dia:
        cont_d[10] += 1
    if "20200112" == dia:
        cont_d[11] += 1
    if "20200113" == dia:
        cont_d[12] += 1
    if "20200114" == dia:
        cont_d[13] += 1
    if "20200115" == dia:
        cont_d[14] += 1
    if "20200116" == dia:
        cont_d[15] += 1
    if "20200117" == dia:
        cont_d[16] += 1
    if "20200118" == dia:
        cont_d[17] += 1
    if "20200119" == dia:
        cont_d[18] += 1
    if "20200120" == dia:
        cont_d[19] += 1
    if "20200121" == dia:
        cont_d[20] += 1
    if "20200122" == dia:
        cont_d[21] += 1
    if "20200123" == dia:
        cont_d[22] += 1
    if "20200124" == dia:
        cont_d[23] += 1
    if "20200125" == dia:
        cont_d[24] += 1
    if "20200126" == dia:
        cont_d[25] += 1
    if "20200127" == dia:
        cont_d[26] += 1
    if "20200128" == dia:
        cont_d[27] += 1
    if "20200129" == dia:
        cont_d[28] += 1
    if "20200130" == dia:
        cont_d[29] += 1
    if "20200131" == dia:
        cont_d[30] += 1
    if "20200209" == dia:
        cont_d[31] += 1
    if "20200310" == dia:
        cont_d[32] += 1
    if "20200413" == dia:
        cont_d[33] += 1
    if "20200630" == dia:
        cont_d[34] += 1
    
    if "00" == hora:
        cont_h[0] += 1
    if "01" == hora:
        cont_h[1] += 1
    if "02" == hora:
        cont_h[2] +=1
    if "03" == hora:
        cont_h[3] +=1
    if "04" == hora:
        cont_h[4] +=1
    if "05" == hora:
        cont_h[5] +=1
    if "06" == hora:
        cont_h[6] +=1
    if "07" == hora:
        cont_h[7] +=1
    if "08" == hora:
        cont_h[8] +=1
    if "09" == hora:
        cont_h[9] +=1
    if "10" == hora:
        cont_h[10]+=1
    if "11" == hora:
        cont_h[11]+=1
    if "12" == hora:
        cont_h[12]+=1
    if "13" == hora:
        cont_h[13]+=1
    if "14" == hora:
        cont_h[14]+=1
    if "15" == hora:
        cont_h[15]+=1
    if "16" == hora:
        cont_h[16]+=1
    if "17" == hora:
        cont_h[17]+=1
    if "18" == hora:
        cont_h[18]+=1
    if "19" == hora:
        cont_h[19]+=1
    if "20" == hora:
        cont_h[20]+=1
    if "21" == hora:
        cont_h[21]+=1
    if "22" == hora:
        cont_h[22]+=1
    if "23" == hora:
        cont_h[23]+=1

#    alltext = alltext + " " + row[4]

plt.figure()
obj1 = ("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23")
y = np.arange(len(obj1))
x = cont_h
plt.bar(y,x)
plt.xticks(np.arange(len(obj1)),obj1,rotation=20)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(0, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma de liga????es em rela????o as horas do dia")
plt.ylabel("Quantidade de Liga????es")
plt.xlabel("Hora")

plt.figure()
obj2 = ("0101","0102","0103","0104","0105","0106","0107","0108","0109","0110","0111","0112","0113","0114","0115","0116","0117","0118","0119","0120","0121","0122","0123","0124","0125","0126","0127","0128","0129","0130","0131","0209","0310","0413","0630")
y = np.arange(len(obj2))
x = cont_d
plt.bar(y,x)
plt.xticks(np.arange(len(obj2)),obj2,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(0, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma de liga????es em rela????o aos dias")
plt.xlabel("Dia")
plt.ylabel("Quantidade de Liga????es")

alltext="reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) reduz(24) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) horario(39) deslig(39) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) maquin(50) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12) tensao(12) conversor(12)  usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) usina(15) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) mega(18) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) geracao(22) compensador(9) compensador(9) compensador(9) compensador(9) compensador(9) compensador(9) compensador(9) compensador(9) compensador(9) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10) barra(10) milimetro(10)  intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) intervencao(11) potencia(11) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) chuva(17) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21) mw(21) sgi(21)  compensa(4) vazao(4) watt(4) compensa(4) vazao(4) watt(4) compensa(4) vazao(4) watt(4) compensa(4) vazao(4) watt(4) manobr(6) reservatorio(6) manobr(6) reservatorio(6) manobr(6) reservatorio(6) manobr(6) reservatorio(6) manobr(6) reservatorio(6) manobr(6) reservatorio(6)  montante(3) montante(3) montante(3) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) disjuntor(16) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) kv(19) gerar(5) gerar(5) gerar(5) gerar(5) gerar(5) documento(2) prorrog(2) vertimento(2) documento(2) prorrog(2) vertimento(2) metro(7) metro(7) metro(7) metro(7) metro(7) metro(7) metro(7) elev(8) elev(8) elev(8) elev(8) elev(8) elev(8) elev(8) elev(8) nivel(8) nivel(8) nivel(8) nivel(8) nivel(8) nivel(8) nivel(8) nivel(8)"
for color in ['Dark2', 'prism']:
    wordcloud = WordCloud(max_font_size=25, 
                          background_color='white',
                          contour_color='white',
                          colormap=color).generate(alltext)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(str(color)+'.png', dpi=300, facecolor='w', edgecolor='w',
       orientation='landscape', format='png')
    plt.show()
    

passos_dur=np.zeros(6)
for dur in duracao:
    if dur>0 and dur<=10:
        passos_dur[0]+=1
    elif dur>10 and dur<=20:
        passos_dur[1]+=1
    elif dur>20 and dur<=30:
        passos_dur[2]+=1
    elif dur>30 and dur<=40:
        passos_dur[3]+=1
    elif dur>40 and dur<=50:
        passos_dur[4]+=1
    else:
        passos_dur[5]+=1
        
plt.figure()
obj = ("[0,10]","]10,20]","]20,30]","]30,40]","]40,50]",">50")
y = np.arange(len(obj))
x = passos_dur
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(0, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma da dura????o das liga????es")
plt.xlabel("Faixa (s)")
plt.ylabel("Quantidade de Liga????es")

print(sum(duracao) / len(duracao))