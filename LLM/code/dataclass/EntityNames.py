import pandas as pd
import os


domain_dict = {
    'lastfm': 'music',
    'dbbook': 'book',
    'movielens': 'movie',
    'boardgamegeek': 'boardgame'
}


def __get_movielens(datapath):
    def fix_title(title):
        if ", The (" in title:
            name_film, _, year = title.rpartition(", The (")
            title = "The " + name_film + " (" + year
            return title
        if ", A (" in title:
            name_film, _, year = title.rpartition(", A (")
            title = "A " + name_film + " (" + year
        return title
    df_movies = pd.read_csv(os.path.join(datapath, r"movies.dat"), sep="::", names=["item_id", "name", "geners"], encoding='ISO-8859-1')
    df_movies['name'] = df_movies['name'].apply(fix_title)
    import re

    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_entities.tsv'), \
            names=['url', 'id_set'], sep='\t')
    dbpedia_mapping = pd.read_csv(os.path.join(datapath, 'MappingMovielens2DBpedia-1.2.tsv'), \
            names=['id_movie', 'name', 'dbpedia_url'], sep='\t')
    df_movies = dbpedia_mapping.set_index('dbpedia_url').join(df_relations.set_index('url')).reset_index()
    df_movies['name'] = df_movies['name'].apply(lambda x: re.sub("\s+\(\d+\)$", "", x))
    df_movies['name'] = df_movies['name'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x).strip())
    df_movies.dropna(inplace=True)
    df_movies['id_set'] = df_movies['id_set'].astype(int)
    mapping_dict = df_movies.loc[:, ['id_set', 'name']].set_index('id_set').to_dict()['name']
    return mapping_dict

def __get_dbbook(datapath):
    user_item_df_train = pd.read_csv(os.path.join(datapath, 'user-item', 'train.tsv'), \
            names=['user', 'item', 'rating'], sep='\t')
    user_item_df_test = pd.read_csv(os.path.join(datapath, 'user-item', 'test.tsv'), \
                names=['user', 'item', 'rating'], sep='\t')
    df = pd.concat([user_item_df_train, user_item_df_test], ignore_index=True)
    all_items = df['item'].unique()
    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_entities.tsv'), sep='\t')
    df_relations['name'] = df_relations['uri'].apply(lambda x: x.split(";")[0])
    temp = df_relations[df_relations['id'].isin(all_items)]
    mapping_dict = temp.set_index('id').to_dict()['name']
    return mapping_dict

def __get_lastfm(datapath):
    df_relations = pd.read_csv(os.path.join(datapath, 'mapping_items.tsv'), \
            names=['id', 'name'], sep='\t')
    mapping_dict = df_relations.set_index('id').to_dict()['name']
    return mapping_dict

def __get_boardgamegeek(datapath):
    df_relations = pd.read_csv(os.path.join(datapath, 'mapped_entities.csv'), sep=';')
    df_relations['name'] = df_relations['primary']
    mapping_dict = df_relations.set_index('id').to_dict()['name']
    return mapping_dict

def set_dataset(dataset_config):
    dataset = dataset_config.get('name', '')
    domain = domain_dict[dataset]
    datapath = dataset_config.get('data_folder', '')
    if dataset=='movielens':
        mapping_dict = __get_movielens(datapath)
    elif dataset=='dbbook':
        mapping_dict = __get_dbbook(datapath)
    elif dataset=='lastfm':
        mapping_dict = __get_lastfm(datapath)
    elif dataset=='boardgamegeek':
        mapping_dict = __get_boardgamegeek(datapath)
    else:
        raise ValueError("Unknown Dataset")
    return list(mapping_dict.items())