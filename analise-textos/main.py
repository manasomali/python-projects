import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

new_file = csv.reader(open('input/Pasta1.csv', 'r'),delimiter='_')
cont_h = np.zeros(24)
cont_d = np.zeros(35)
alltext = ""
for row in tqdm(list(new_file)):
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

    alltext = alltext + " " + row[4]

plt.figure()
obj1 = ("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23")
y = np.arange(len(obj1))
x = cont_h
plt.bar(y,x)
plt.xticks(np.arange(len(obj1)),obj1,rotation=20)
plt.title("Histograma de ligações em relação as horas do dia")
plt.ylabel("Quantidade de Ligações")
plt.xlabel("Hora")

plt.figure()
obj2 = ("0101","0102","0103","0104","0105","0106","0107","0108","0109","0110","0111","0112","0113","0114","0115","0116","0117","0118","0119","0120","0121","0122","0123","0124","0125","0126","0127","0128","0129","0130","0131","0209","0310","0413","0630")
y = np.arange(len(obj2))
x = cont_d
plt.bar(y,x)
plt.xticks(np.arange(len(obj2)),obj2,rotation=90)
plt.title("Histograma de ligações em relação aos dias")
plt.xlabel("Dia")
plt.ylabel("Quantidade de Ligações")


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