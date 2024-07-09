from IPython.display import display
import pandas as pd 
import numpy as np

data = pd.read_csv('BTH2.csv') 
#doc du lieu tu bang
display(data)
#hien thi gia tri trong bang
data.info()
#thong tin cua bang
data.value_counts()
#thong ke so luong gia tri
data.isna().sum()
#thong ke so luong gia tri con thieu
num_sample = 5
data.sample(num_sample)
#lay so luong mau du lieu
df1 = data.copy()
#copy du lieu

Phương Pháp Giá Trị Trung Bình:
data.fillna() 
#hàm điền giá trị NaN
data.round() 
#hàm tròn giá trị cho trước
data.mean() 
#hàm lấy giá trị trung bình
df1['Age'].fillna(round(df1['Age'].mean()),inplace=True)
df1['Income'].fillna(round(df1['Income'].mean()),inplace=True)

Phương Pháp Tần Số Cao Nhất:
data.mode() 
#hàm lấy giá trị có tần số xuất hiện lớn nhất
df1['Region'].fillna(df1['Region'].mode()[0], inplace = True)

Phương Pháp Chuẩn Hóa Min Max:

df2=df1.copy()
age_min=df2['Age'].min()
age_max=df2['Age'].max()
income_min = df2['Income'].min()
income_max = df2['Income'].max()
#nhap gia tri

df2['Age'] = (df2['Age'] - age_min)/(age_max - age_min)
df2['Income'] = (df2['Income'] - income_min)/(income_max - income_min)
#cong thuc

Phương Pháp Liên Tục Hóa Dữ Liệu:

df3['Region'].unique()
#tra ve gia tri 1 0
dummies = pd.get_dummies(df3['Region'])
#tra ve du lieu dang cot
df3 = pd.concat([df3,dummies], axis=1) #axis=1 : ghep hang theo chieu ngang
df3.drop('Region', axis=1, inplace=True)  #drop : loai bo hang khoi bang du lieu
#ghep du lieu dang cot voi bang co san

Phương Pháp Rời Rạc Hóa Dữ Liệu:

df4.Income = pd.cut(df4["Income"], #cut : roi rac hoa du lieu
       bins=[-0.1, 0.2, 0.4, 0.6, 0.8, 1], # cac khoang gia tri
       labels=["Very Low", "Low", "Medium", "High", "Very High"]) # gia tri tra ve

Phương Pháp Quy Hồi Tuyến Tính

df5['Online Shopper'].replace({"No":0,"Yes":1},inplace=True) 
#chuyen hoa du lieu 

Train = df5.drop("Online Shopper", axis=1) 
Test = df5["Online Shopper"]
display(Train)
display(Test)

from sklearn.linear_model import RidgeClassifier
tp = int(0.9 * len(Train)) #thread point

X_train = Train[:tp]
Y_train = Test[:tp]
X_test = Train[tp:]
Y_test = Test[tp:]
#tao nen bo test va train
reg = RidgeClassifier().fit(X_train, Y_train)
pred = reg.predict(X_test)
#huan luyen du lieu
reg.score(X_test,Y_test)
#tinh toan do chinh xac
target_class = pd.DataFrame({'Test values': pred,'True value': Y_test})
# so sanh ket qua train va test


VISUALIZATION :

data.value_counts()
#chuyển hóa số liệu lớn
data.to_numpy()
#chuyển số liệu thành mảng
data.to_frame().index.values
#chuyển số liệu cột chính sang mảng