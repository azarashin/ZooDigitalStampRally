import json

class Filter:
    def __init__(self):
        with open("animal_data_set.json", "r", encoding="utf-8") as f:
            self._data_list = json.load(f)

        facility_list = []
        for d in self._data_list:
            for facility in d['facilities']:
                facility_list.append(facility)
        self._facility_list = list(set(facility_list))
        self._facility_list.sort()

    def run(self):
        while True:
            print('--- menu ---')
            print('1. 動物園の一覧を表示する')
            print('2. 指定された動物園にいる動物の一覧を表示する')
            print('999. 終了する')
            id = input('input: ').strip()
            if id == '1':
                self.show_facility_list()
            if id == '2':
                self.filter_place()
            if id == '999':
                exit()
    
    def show_facility_list(self):
        for facility in self._facility_list:
            print(facility)

    def filter_place(self):
        for i in range(0, len(self._facility_list)):
            facility = self._facility_list[i]
            print(f'{i}: {facility}')
        id = int(input('input: ').strip())
        facility = self._facility_list[id]
        animals = [d for d in self._data_list if facility in d['facilities']]
        path = f'animal_list_in_place[{id}].tsv'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'日本語名\t学名\tImageURL\tTAXON_ID\n')
            for animal in animals:
                print(f'{animal['name']}\t{animal['image_url']}')
                f.write(f'{animal['name']}\t\t{animal['image_url']}\t\n')
        print(f'計：{len(animals)}')
        print(f'{path} に一覧が出力されました')

filter =  Filter()
filter.run()


