import requests
from bs4 import BeautifulSoup
import json
import os

# スクレイピングされたHTMLから必要な情報を抽出する

image_header = '../Encyclopedia/Species'
image_base_url = 'https://www.tokyo-zoo.net/Encyclopedia/Species/'

data_list = []

for id in range(0, 1000):
    path = f'cache/{id}.html'
    if not os.path.exists(path):
        continue
    html_content = open(path, "r", encoding="utf-8").read() 

    soup = BeautifulSoup(html_content, 'html.parser')

    # 情報を抽出する
    data = {}

    # 名称の抽出
    name_tag = soup.find('td', string='名称')
    name_value = name_tag.find_next('td') if name_tag else None
    data['名称'] = name_value.text.strip() if name_value else '情報なし'

    # 飼育園館の抽出
    facility_tag = soup.find('td', string='飼育園館')
    facility_value = facility_tag.find_next('td') if facility_tag else None
    data['飼育園館'] = facility_value.text.strip() if facility_value else '情報なし'

    # 生息地の抽出
    habitat_tag = soup.find('td', string='生息地')
    habitat_value = habitat_tag.find_next('td') if habitat_tag else None
    data['生息地'] = habitat_value.text.strip() if habitat_value else '情報なし'

    # 体の大きさの抽出
    size_tag = soup.find('td', string='体の大きさ')
    size_value = size_tag.find_next('td') if size_tag else None
    data['体の大きさ'] = size_value.text.strip() if size_value else '情報なし'

    # えさの抽出
    food_tag = soup.find('td', string='えさ')
    food_value = food_tag.find_next('td') if food_tag else None
    data['えさ'] = food_value.text.strip() if food_value else '情報なし'

    # 特徴の抽出
    feature_tag = soup.find('td', string='特徴')
    feature_value = feature_tag.find_next('td') if feature_tag else None
    data['特徴'] = feature_value.text.strip() if feature_value else '情報なし'

    images = soup.find_all('img', src=lambda x: x and x.startswith(image_header))
    image_url = image_base_url + images[0]['src'][len(image_header):] if len(images) > 0 else ''
    print(image_url)

    data_list.append({
        'name': data['名称'], 
        'image_url': image_url, 
        'facilities': data['飼育園館'].split('・'), 
        'habitat': data['生息地'], 
        'size': data['体の大きさ'], 
        'food': data['えさ'], 
        'feature': data['特徴'], 
    })

with open("animal_data_set.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)