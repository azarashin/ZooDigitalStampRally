import requests
import time

# Webサイトから情報をスクレイピングにより収集する

interval_sec = 1

for id in range(0, 1000):
    url = f'https://www.tokyo-zoo.net/encyclopedia/species_detail?species_code={id}'

    response = requests.get(url)

    if response.status_code == 200:
        with open(f'cache/{id}.html', 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f'ok({id})')
    else:
        print(f'ng({id})')
    time.sleep(interval_sec)
