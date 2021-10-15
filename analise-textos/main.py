import csv
from tqdm import tqdm
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

file1 = csv.reader(open('input/dataset_original.csv', 'r'),delimiter='_')
cont_h = np.zeros(24)
cont_d = np.zeros(35)
alltext = ""
duracao_original = []
duracao_silenciado = []
for row in tqdm(list(file1)):
    file2 = csv.reader(open('input/dataset_original.csv', 'r'),delimiter='_')
    for row2 in list(file2):
        if row[0]== row2[0]:
            duracao_original.append(float(row2[4]))
            break
    
    file3 = csv.reader(open('input/dataset_silenciado.csv', 'r'),delimiter='_')
    for row3 in list(file3):
        if row[0]== row3[0]:
            duracao_silenciado.append(float(row3[4]))
            break
        
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

plt.figure(figsize=(8, 6), dpi=300)
obj1 = ("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23")
y = np.arange(len(obj1))
x = cont_h
plt.bar(y,x)
plt.xticks(np.arange(len(obj1)),obj1,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(1, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma de ligações em relação as horas do dia")
plt.ylabel("Quantidade de Ligações")
plt.xlabel("Hora")


plt.figure(figsize=(8, 6), dpi=300)
obj1 = ("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23")
x = np.arange(len(obj1))
y = cont_h
y2 = [13840, 13019, 12361, 11863, 11629, 11579, 11544, 12077, 13856, 15226, 15869, 16526, 16246, 15687, 17224, 17364, 16002, 16312, 15116, 13745, 14744, 14768, 15062, 14751]
plt.bar(x,y)
plt.xticks(np.arange(len(obj1)),obj1,rotation=90)
plt.title("Histograma de ligações em relação as horas do dia e carga da região sul")
plt.ylabel("Quantidade de Ligações", color='tab:blue')
plt.xlabel("Hora")
axes2 = plt.twinx()
axes2.plot(x, y2, color='tab:red')
axes2.set_ylabel('Carga Sul (MW)', color='tab:red')


plt.figure(figsize=(8, 6), dpi=300)
obj2 = ("01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35")
y = np.arange(len(obj2))
x = cont_d
plt.bar(y,x)
plt.xticks(np.arange(len(obj2)),obj2,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(1, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma de ligações em relação aos dias")
plt.xlabel("Dia")
plt.ylabel("Quantidade de Ligações")

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
    

passos_dur_original=np.zeros(7)
for dur in duracao_original:
    if dur>0 and dur<=10:
        passos_dur_original[0]+=1
    elif dur>10 and dur<=20:
        passos_dur_original[1]+=1
    elif dur>20 and dur<=30:
        passos_dur_original[2]+=1
    elif dur>30 and dur<=40:
        passos_dur_original[3]+=1
    elif dur>40 and dur<=50:
        passos_dur_original[4]+=1
    elif dur>50 and dur<=60:
        passos_dur_original[5]+=1
    else:
        passos_dur_original[6]+=1
        
passos_dur_silenciado=np.zeros(7)
for dur in duracao_silenciado:
    if dur>0 and dur<=10:
        passos_dur_silenciado[0]+=1
    elif dur>10 and dur<=20:
        passos_dur_silenciado[1]+=1
    elif dur>20 and dur<=30:
        passos_dur_silenciado[2]+=1
    elif dur>30 and dur<=40:
        passos_dur_silenciado[3]+=1
    elif dur>40 and dur<=50:
        passos_dur_silenciado[4]+=1
    elif dur>50 and dur<=60:
        passos_dur_silenciado[5]+=1
    else:
        passos_dur_silenciado[6]+=1
        
plt.figure(figsize=(8, 6), dpi=300)
obj = ("[0,10]","]10,20]","]20,30]","]30,40]","]40,50]","]50,60]",">60")
y = np.arange(len(obj))
x = passos_dur_original
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(-0.7, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma da duração das ligações")
plt.xlabel("Faixa (s)")
plt.ylabel("Quantidade de Ligações")

print("média de duração das ligações original: {:f}".format(sum(duracao_original) / len(duracao_original)))

plt.figure(figsize=(8, 6), dpi=300)
obj = ("[0,10]","]10,20]","]20,30]","]30,40]","]40,50]","]50,60]",">60")
y = np.arange(len(obj))
x = passos_dur_silenciado
plt.bar(y,x)
plt.xticks(np.arange(len(obj)),obj,rotation=90)
plt.axhline(y=x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(-0.7, x.mean(),'Mean: {:.0f}'.format(x.mean()))
plt.title("Histograma da duração sem silêncio das ligações")
plt.xlabel("Faixa (s)")
plt.ylabel("Quantidade de Ligações")

print("média de duração das ligações sem silêncio: {:f}".format(sum(duracao_silenciado) / len(duracao_silenciado)))