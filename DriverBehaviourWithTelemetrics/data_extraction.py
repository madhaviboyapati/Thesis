import json
import pandas as pd
import os
import csv
from pandas.io.json import json_normalize #package for flattening json in pandas df
root =os.getcwd()

for root,dirs,files in os.walk(os.path.join(root,"car_data")):
    for subdirs in dirs:
        for root2,dirs2,files2 in os.walk(os.path.join(root,subdirs)):
            for subdirs2 in dirs2:

                for root3, dirs3, files3 in os.walk(os.path.join(root2,subdirs2)):
                    df_write=pd.DataFrame()
                    for each in files3:
                        with open(os.path.join(root3,each), 'rb') as f:
                            d = json.load(f)
                            f
                            lat_d = pd.json_normalize(d['truck'])
                            df_write = df_write.append(flat_d, ignore_index=True)
                    filename = subdirs2
                    file = filename + ".csv"
                    df_write.to_csv(os.path.join(root3,file),sep=',')
                   # with open(os.path.join(root3,file)) as w:




#lets put the data into a pandas df
#clicking on raw_nyc_phil.json under "Input Files"
#tells us parent node is 'programs'




#pd.write_csv('data.csv', flat_d)
