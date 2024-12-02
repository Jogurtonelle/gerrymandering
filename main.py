"""
"properties": {
        "gml_id": "",
        "JPT_SJR_KO": "GMI",
        "JPT_POWIER": "4889",
        "JPT_KOD_JE": "3215011",
        "JPT_NAZWA_": "Szczecinek",
        "JPT_ORGAN_": "",
        "JPT_JOR_ID": "13151",
        "WERSJA_OD": "20120926",
        "WERSJA_DO": "0",
        "WAZNY_OD": "20120926",
        "WAZNY_DO": "0",
        "JPT_KOD__1": "",
        "JPT_NAZWA1": "",
        "JPT_ORGAN1": "NZN",
        "JPT_WAZNA_": "NZN",
        "ID_BUFORA_": "13616",
        "ID_BUFORA1": "0",
        "ID_TECHNIC": "828110",
        "IIP_PRZEST": "PL.PZGIK.200",
        "IIP_IDENTY": "f3535d28-7251-4b68-b534-c815f3f89259",
        "IIP_WERSJA": "2012-09-26T22:34:52+02:00",
        "JPT_KJ_IIP": "EGIB",
        "JPT_KJ_I_1": "3215011",
        "JPT_KJ_I_2": "",
        "JPT_OPIS": "",
        "JPT_SPS_KO": "UZG",
        "ID_BUFOR_1": "0",
        "JPT_ID": "828110",
        "JPT_POWI_1": "0",
        "JPT_KJ_I_3": "",
        "JPT_GEOMET": "0",
        "JPT_GEOM_1": "0",
        "SHAPE_LENG": ".5910644718",
        "SHAPE_AREA": ".00659579547",
        "REGON": "33092090800000",
        "RODZAJ": "gmina"
      }"""

import geopandas as gpd
import pandas as pd
import pickle as pkl
import networkx as nx
import json

# with open('gminy.geojson', encoding="UTF-8") as gminy:
# 	gmin = gpd.read_file(gminy)
# 	print(len(gmin['JPT_KOD_JE'].tolist()))


# gdf =  gpd.read_file("shapefiles/A03_Granice_gmin.shp", encoding="utf-8")

# gdf['JPT_KOD_JE'] = gdf['JPT_KOD_JE'].str[:-1]

# # for index, gmina in gdf.iterrows():   
# 	# # get 'not disjoint' countries
# 	# neighbors = gdf[~gdf.geometry.disjoint(gmina.geometry)].JPT_KOD_JE.tolist()

# 	# # remove own name of the country from the list
# 	# neighbors = [ id for id in neighbors if gmina.JPT_KOD_JE != id ]

# 	# # add names of neighbors as NEIGHBORS value
# 	# gdf.at[index, "NEIGHBORS"] = ", ".join(neighbors)
	
# gdf.drop(columns=['JPT_SJR_KO', 'JPT_POWIER', 'JPT_ORGAN_', 'JPT_JOR_ID', 'WERSJA_OD', 'WERSJA_DO', 'WAZNY_OD', 'WAZNY_DO', 'JPT_KOD__1', 'JPT_NAZWA1', 'JPT_ORGAN1', 'JPT_WAZNA_', 'ID_BUFORA_', 'ID_BUFORA1', 'ID_TECHNIC', 'IIP_PRZEST', 'IIP_IDENTY', 'IIP_WERSJA', 'JPT_KJ_IIP', 'JPT_KJ_I_1', 'JPT_KJ_I_2', 'JPT_OPIS', 'JPT_SPS_KO', 'ID_BUFOR_1', 'JPT_ID', 'JPT_POWI_1', 'JPT_KJ_I_3', 'JPT_GEOMET', 'JPT_GEOM_1', 'Shape_Leng', 'Shape_Area', 'REGON'], inplace=True)

# gdf.to_file("gminy_polska_2024.shp", encoding="utf-8")



# gdf =  gpd.read_file("gminy_polska_2024.shp", encoding="utf-8")

# graph = nx.Graph()

# for index, gmina in gdf.iterrows():
# 	#Get a list of neighboring counties' IDs (TERC)
# 	neighbors = gdf[~gdf.geometry.disjoint(gmina.geometry)].JPT_KOD_JE.tolist()
# 	#Remove own ID of the county from the list
# 	neighbors = [ id for id in neighbors if gmina.JPT_KOD_JE != id ]

# 	#Add edges to the graph
# 	for neighbor in neighbors:
# 		graph.add_edge(gmina['JPT_KOD_JE'], neighbor)

# pkl.dump(graph, open("graph.pkl", "wb"))

#graph = pkl.load(open("graph.pkl", "rb"))

df = pd.read_csv("election_data_pl/wyniki_gl_na_listy_po_gminach_sejm_utf8.csv", encoding="utf-8", delimiter=";")
#pint first row
print(df.iloc[0])

#covert every float to int
df = df.fillna(0)
df = df[df.columns[5:]].astype(int)

print(df.iloc[0])
