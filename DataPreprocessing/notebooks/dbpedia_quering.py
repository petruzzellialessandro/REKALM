import json
import pickle
import re
import requests
from tqdm import tqdm
import time
import pandas as pd

dataset = 'lastfm'
types_prop = ['rdfs:label', 'dbp:name', 'dbp:title']
domain = {
    'lastfm': 'music',
    'dbbook': 'book',
    'movielens': 'movie'
}[dataset]
import os
datapath = os.path.join('..', 'data', dataset)


def parse_name_from_url(url):
    name = url.split("/")[-1]
    name = name.replace("_", " ")
    name = name.split(':')[-1]
    name = re.sub(r"\([^()]*\)", "", name)
    pattern = r",.*$"
    name = re.sub(pattern, "", name)
    return name


def prepare_query(url, typ):
    query = f"""
    SELECT ?name WHERE {{
        <{url}> {typ} ?name .
    }}
    """
    return {"query": query}
      
# Define the data for the HTTP request

def get_names_dpedia(id_urls, types_prop):
    # Define the DBpedia SPARQL endpoint
    endpoint_url = "https://dbpedia.org/sparql"

    # Define the headers for the HTTP request
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    # Extract the resource from the URL
    names = {}
    for id_, url in id_urls.items():
        url = url.strip("<>")
        url = url.split("__PersonFunction__1")[0]
        while True:
            for typ in types_prop:
                data = prepare_query(url, typ)
            try:
                # Send the HTTP request
                response = requests.post(endpoint_url, headers=headers, data=data)

                # Parse the JSON response
                results = json.loads(response.text)

                # Check if an abstract was found
                if results["results"]["bindings"]:
                    # Extract the abstract from the results
                    name = results["results"]["bindings"][0]["name"]["value"]
                    names[id_] = name
                    break
                else:
                    names[id_] = parse_name_from_url(url)
            except requests.exceptions.RequestException:
                # Wait for 30 seconds before trying again
                time.sleep(30)
            except Exception:
                continue
            break
    return names

if dataset=='movielens':
    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_entities.tsv'), \
            names=['url', 'id'], sep='\t')
    df_relations = df_relations[df_relations['url'].str.contains('http')]

if dataset=='dbbook':
    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_entities.tsv'), sep='\t')
    df_relations['name'] = df_relations['uri'].apply(lambda x: x.split(";")[0])
    df_relations['url'] = df_relations.apply(lambda row: row['uri'].replace(row['name'], '').replace(';', '') if row['uri']!=row['name'] else row['uri'], axis=1)

if dataset=='lastfm':
    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_props.tsv'), \
            names=['url', 'id'], sep='\t')
    
def process_in_chunks(url_dic, types_prop, dir_path, chunk_size=500):
    result_dict = {}
    keys = list(url_dic.keys())
    start = 0
    for i in tqdm(range(start, len(keys), chunk_size)):
        chunk_keys = keys[i : i + chunk_size]
        chunk = dict((k,url_dic[k]) for k in chunk_keys)
        chunk_dict = get_names_dpedia(chunk, types_prop)
        result_dict = {**result_dict, **chunk_dict}
        print("Saving partial results up to", i + chunk_size)
        with open(dir_path + f"entities_names_{start}_{str(i + chunk_size)}.pickle", "wb") as f:
            pickle.dump(result_dict, f)
    return result_dict

entities_dic = df_relations.set_index('id').to_dict()['url']
entities_abstracts = process_in_chunks(entities_dic, types_prop, datapath, chunk_size=500)
with open(datapath + "entities_names_.pickle", "wb") as f:
    pickle.dump(entities_abstracts, f)