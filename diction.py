info={
    'stu1101':[12,23,4,5],
    'stu1102':"baidu",
    'stu1103':"ali",
}
print('stu1101' in info)
a=info.items()
print(a)
c=info.fromkeys([6,7,8],"test")
info[-1]='asdas'
print(info['stu1101'])

m=range(8)
n=m[::4]