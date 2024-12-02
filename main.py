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

df = pd.read_csv("election_data_pl/wyniki_gl_na_listy_po_gminach_sejm_utf8_copy.csv", encoding="utf-8", delimiter=",")
# #pint first row

# #covert every float to int
# df[df.columns] = df[df.columns].fillna(0)
# df[df.columns[3:]] = df[df.columns[3:]].astype(int)
# df[df.columns[0]] = df[df.columns[0]].astype(int)
# df[df.columns[0]] = df[df.columns[0]].astype(str)
# df.drop(columns=['Powiat'], inplace=True)
# df['TERYT Gminy'] = df['TERYT Gminy'].apply(lambda x: '0' + str(x) if len(str(x)) == 5 else str(x))
# df['TERYT Gminy'] = df['TERYT Gminy'].apply(lambda x: '146501' if x[:-2] == '1465' else x)

# df.to_csv("election_data_pl/wyniki_gl_na_listy_po_gminach_sejm_utf8_copy.csv", encoding="utf-8", index=False)

graph = pkl.load(open("graph.pkl", "rb"))

#add number of voters to the graph
for node in graph.nodes:
	graph.nodes[node]['voters'] = 0

for index, row in df.iterrows():
	county_id = str(row.at['TERYT Gminy'])
	if county_id == '0':
		county_id = '146501' #ID Warszawy
	elif len(county_id) == 5:
		county_id = '0' + county_id
	
	graph.nodes[county_id]['voters'] += row.at['Liczba wyborców uprawnionych do głosowania']
	
import districting_methods
districting_methods.seeding_algorithm(graph, 460)

# gdf = gpd.read_file("shapefiles/gminy_polska_2024.shp", encoding="utf-8")

# for index, row in df.iterrows():
# 	county_id = str(row.at['TERYT Gminy'])
# 	if county_id == 'nan':
# 		county_id = '146501' #ID Warszawy
# 	#cut .0 from the end of the string
# 	if county_id[-2:] == '.0':
# 		county_id = county_id[:-2]
# 	if len(county_id) == 5:
# 		county_id = '0' + county_id

# 	#get index of collumn [5:] with the highest number of votes
# 	max_votes = row.iloc[5:].idxmax()
# 	#add new column to gdf with the name of the party that won the elections in the county
# 	gdf.loc[gdf['JPT_KOD_JE'] == county_id, 'WINNER'] = max_votes

#import matplotlib.pyplot as plt
##plot the map to png
#let me set colors for each party
#KOMITET WYBORCZY BEZPARTYJNI SAMORZĄDOWCY - szary
#KOMITET WYBORCZY WYBORCÓW - zielony
#KOMITET WYBORCZY PRAWO I SPRAWIEDLIWOŚĆ - niebieski
#KOMITET WYBORCZY KOALICJA OBYWATELSKA - czerwony
#KOMITET WYBORCZY POLSKIE STRONNICTWO LUDOWE - żółty

# def get_color(party):
# 	if "BEZPARTYJNI SAMORZĄDOWCY" in party:
# 		return 'grey'
# 	elif "KOALICJA OBYWATELSKA" in party:
# 		return 'orange'
# 	elif "PRAWO I SPRAWIEDLIWOŚĆ" in party:
# 		return 'blue'
# 	elif "LEWICA" in party:
# 		return 'red'
# 	elif "POLSKIE STRONNICTWO LUDOWE" in party:
# 		return 'green'
# 	else:
# 		return 'yellow'

# gdf['color'] = gdf['WINNER'].apply(get_color)
# gdf.plot(color = gdf['color'], figsize=(20, 20), label=gdf['WINNER'])
# plt.savefig("map.png")

# plt.plot()
# plt.figure(figsize=(10, 10))
# for index, row in gdf.iterrows():
# 	if "BEZPARTYJNI SAMORZĄDOWCY" in row['WINNER']:
# 		color = 'grey'
# 	elif "KOALICJA OBYWATELSKA" in row['WINNER']:
# 		color = 'orange'
# 	elif "PRAWO I SPRAWIEDLIWOŚĆ" in row['WINNER']:
# 		color = 'blue'
# 	elif "LEWICA" in row['WINNER']:
# 		color = 'red'
# 	elif "POLSKIE STRONNICTWO LUDOWE" in row['WINNER']:
# 		color = 'green'
# 	else:
# 		color = 'black'
# 	gdf[gdf['JPT_KOD_JE'] == row['JPT_KOD_JE']].plot(ax=plt.gca(), color=color, edgecolor='black', label=row['WINNER'])
# plt.legend(loc='upper left')

