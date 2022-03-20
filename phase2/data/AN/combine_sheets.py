import os
import pandas as pd


writer = pd.ExcelWriter('AN_data_collection.xlsx')

dir_path = './sheets/'
for sheet in os.listdir(dir_path):
    print(sheet)

    df = pd.read_excel(dir_path+sheet, header=1)
    df = df.iloc[:, 18:-2]
    print(df)

    df.to_excel(writer, sheet_name=sheet.split('_')[1], index=None)

writer.save()
