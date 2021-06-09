# f=open('ly.txt','r',encoding='utf-8') # 文件句柄 'w'为创建文件，之前的数据就没了
# data=f.read()
# print(data)
# f.close()
#
# f=open('test','a',encoding='utf-8') # 文件句柄 'a'为追加文件 append
# f.write("\n阿斯达所，\n天安门上太阳升")
# f.close()
list_1=[1,4,5,7,3,6,7,9,1]
list_1=set(list_1)
list_2=set([2,6,0,6,22,8,4])
list_3=[1,4,6]
list_4=[1,4,6,7]
list_3=set(list_3)
list_4=set(list_4)
print(list_3.issubset(list_4))
print(list_4.issuperset(list_3))
print(list_1&list_2)
print(list_1^list_2)
print(list_1&list_2!={})
list_1.remove(1)
print(list_1)