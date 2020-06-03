import SAME_IMAGE as si

#GetSame(

#imgs_orig=ПАПКА С ИСХОДНЫМИ ИЗОБРАЖЕНИЯМИ ИЛИ МАССИВ ПУТЕЙ К ИСЗОДНЫМ ИЗОБРАЖЕНИЯМ,

#imgs_search_in=ПАПКА С ИЗОБРАЖЕНИЯМИ ДЛЯ СРАВНЕНИЯ ИЛИ МАССИВ ПУТЕЙ К ИЗОБРАЖЕНИЯМИ ДЛЯ СРАВНЕНИЯ,

#dist_multiplier=0.75,scaler={МАКСИМАЛЬНАЯ ДЛИНА/ШИРИНА:[ДЛЯ ЧТО ИСКАТЬ, ДЛЯ ГДЕ ИСКАТЬ],-1 (ВСЁ ОСТАЛЬНОЕ):[ДЛЯ ЧТО ИСКАТЬ, ДЛЯ ГДЕ ИСКАТЬ]},output=False)




# ДЛЯ ПОВЫШЕНИЯ ТОЧНОСТИ ТУТ ИСПОЛЬЗУЕТСЯ ФИЛЬТР ЗНАЧЕНИЙ, ПОЭТОМУ СКОРЕЕ ВСЕГО НОРМАЛЬНО РАБОТАТЬ С ОДНИМ ФАЙЛОМ, ЕСЛИ МНОГО ФАЙЛОВ, С КОТОРЫМ ЕГО НАДО СРАВНИТЬ, НЕ БУДЕТ
# result = si.GetSame(imgs_orig=["what\\im1.png","what\\im2.png","what\\im3.png"],output=False)


result = si.GetSame(scaler={50:[1.1, 1/1.1],-1:[2.46, 0.374]},dist_multiplier=0.75,output=False)
print("")
print("-"*55)
print("dist_multiplier = ",result['dist_multiplier'])
print("scaler = ",result['scaler'])
for k in result['RESULT']:
	print(k)
print("-"*55)


# result = si.GetSameImageAlgo(2)
# print(result)
input()
