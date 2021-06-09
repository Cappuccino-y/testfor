__author__="Alex Li"
names="zhang Gu Xiang Xu"
names=["zhang","Gu","Xiang","Xu"]
# 1.切片
print(names[0],names[1],names[2])
print(names[1:3])  # 顾头不顾尾，切片
print(names[-1]) # 在不知道多长情况下，取最后一个位置
print(names[-1:-3]) # 切片是从左往右，此时不输出
print(names[-3:-1]) # 顾头顾尾，去最后三个
print(names[-2:])  # 取最后两个
print(names[0:3]) # 切片 等价于 print(names[:3])

# 2.追加
names.append("Lei")
print(names)
# 3.指定位置插入
names.insert(1,"Chen") # Gu前面插入
print(names)
# 4.修改
names[2]="Xie"
print(names)
# 5.删除
# 第一种删除方法
names.remove("Chen")
print(names)
# 第二种删除方法
del names[1]
print(names)
# 第三种删除方法
names.pop() # 默认删除最后一个
print(names)
names.pop(1) #删除第二个元素
print(names)
print(names.index("Xu")) # 1
print(names[names.index("Xu")]) #打印出找出的元素值3
# 6.统计
names.append("zhang") #再加一个用于学习统计"zhang"的个数
print(names.count("zhang"))
# 7.排序
names.sort() #按照ASCII码排序
print(names)
names.reverse() # 逆序
print(names)
# 8.合并
names2=[1,2,3,4]
names.extend(names2)
print(names,names2)
# 9.删掉names2
'''del names2'''
print(names2) # NameError: name 'names2' is not defined,表示已删除
# 10.浅copy
names2=names.copy()
print(names,names2) # 此时names2与names指向相同
names[2]="大张"
print(names,names2) # 此时names改变，names2不变
# 11.浅copy在列表嵌套应用
names=[1,2,3,4,["zhang","Gu"],5]
print(names)
names2=names.copy()
names[3]="斯"
names[4][0]="张改"
print(names,names2) # copy为浅copy,第一层copy不变，后面的嵌套全部都变,修改names2与names都一样
# 12.完整克隆
import copy
# 浅copy与深copy
'''浅copy与深copy区别就是浅copy只copy一层，而深copy就是完全克隆'''
names=[1,2,3,4,["zhang","Gu"],5]
# names2=copy.copy(names) # 这个跟列表的浅copy一样
names2=copy.deepcopy(names) #深copy
names[3]="斯"
names[4][0]="张改"
print(names,names2)

# 13.列表循环
for i in names:
    print(i)
print(names[0:-1:2]) # 步长为2进行切片
# 0与-1都可以省略掉
print(names[::2]) # 步长为2进行切片

# 浅拷贝三种方式
person=['name',['a',100]]
p1=copy.copy(person)
p2=person[:]  #其实p2=person[0:-1],0与-1均可以不写
p3=list(person)
print(p1,p2,p3)

print('+'.join(['1','2','3']))