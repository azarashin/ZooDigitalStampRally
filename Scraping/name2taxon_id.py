import requests
import sys

# 日本語名	学名	ImageURL    TAXON_ID
# と並んでいるファイル（1行目はタイトル）から
# 学名を読み込み、学名からTAXON_ID を求めて書き込む。

def name2taxon_id(scientific_name):
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": scientific_name}

    response = requests.get(url, params=params)
    data = response.json()

    if data["total_results"] > 0:
        taxon = data["results"][0]
        print("Scientific name:", taxon["name"])
        print("TAXON_ID:", taxon["id"])
        return taxon["name"], taxon["id"]
    else:
        print("No results found")
    return None, None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python ./name2taxon_id.py (path_to_table_file)')
        exit()
    path = sys.argv[1].strip()
    path_body = path[:-4]
    extention = path[-4:]
    print(path)
    source = open(path, 'r', encoding='utf-8').readlines()
    title_line = source[0]
    data_lines = source[1:]
    out_path = f'{path_body}.taxon_id{extention}'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('日本語名\t学名\tImageURL\tTAXON_ID\n')
        for data in data_lines:
            name, scientific, url = [d.strip() for d in data.split('\t')]
            scientific, taxon_id = name2taxon_id(scientific)
            f.write(f'{name}\t{scientific}\t{url}\t{taxon_id}\n')

