import kagglehub
import pandas as pd
import json
import re

def parse_title_year(title_with_year):
    year_match = re.search(r'\((\d{4})\)', title_with_year)
    year = int(year_match.group(1)) if year_match else None
    title = re.sub(r'\s*\(\d{4}\)', '', title_with_year).strip()
    return title, year

def df_to_json_arrays(df):
    json_array = []
    json_movie_encoder = []
    json_genres = set()  

    for _, row in df.iterrows():
        movie_id = int(row["movieId"])
        title_with_year = row["title"]
        genres_str = row["genres"]

        title, year = parse_title_year(title_with_year)
        genres = genres_str.split('|')

        if year is not None:
            movie_dict = {
                "movieId": movie_id,
                "title": title,
                "year": year,
                "genres": genres
            }
            json_array.append(movie_dict)
            json_movie_encoder.append(movie_id)
            json_genres.update(genres)

    return json_array, json_movie_encoder, list(json_genres)


def write_js_variable(file_name, variable_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"export const {variable_name} = ")
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write(";")

path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")

df = pd.read_csv(path+"/movies.csv")
json_data, json_movie_encoder, json_genres = df_to_json_arrays(df)


write_js_variable('movies.js', 'movies', json_data)
write_js_variable('mlb_classes.js', 'mlb_classes', json_genres)
write_js_variable('movie_encoder.js', 'movie_encoder', json_movie_encoder)


