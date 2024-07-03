 # Load the dataset
import pandas as pd
import pymysql
df = pd.read_csv(r"C:/Project/CropPredictionSystem (3)/CropPredictionSystem/Recommendationsystem/Crop_recommendation.csv", encoding='unicode_escape')
con=pymysql.connect(host="localhost",user="root",password="root",database="crop")
N=df['N']
P=df['P']
K=df['K']
temperature=df['temperature']
humidity=df['humidity']
ph=df['ph']
rainfall=df['rainfall']
label=df['label']
cur=con.cursor()
for i in range(len(label)):
    sql="INSERT INTO recommendation(N,P,K,temperature,humidity,ph,rainfall,label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)";
    values=(N[i],P[i],K[i],temperature[i],humidity[i],ph[i],rainfall[i],label[i])
    cur.execute(sql,values)
    con.commit()
    

