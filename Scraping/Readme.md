# データ収集環境

## 対象動物の一覧抽出

### WebサイトのHTML取得

downloader.py を使用する。

WebサイトからHTMLデータをスクレイピングにより収集する。
アクセスするサイトは下記。

https://www.tokyo-zoo.net/encyclopedia/species_detail?species_code={id}

アクセス頻度はコード中の変数interval_sec で調整すること。
この値を小さくするとアクセス間隔が短くなり、素早くデータを収集できるが、
サーバに負荷をかけてしまう。サーバへ負荷をかけすぎないよう配慮すること。

```py
interval_sec = 1
```

取得したHTMLデータはcache ディレクトリ配下に格納され、

```
cache/{id}.html
```

のように通し番号で名前付けされる。

### HTML からの情報抽出

pickupper.py を使用する。

cache ディレクトリはいかに格納された通し番号付きの名前のHTMLデータを解析し、
下記の情報をjson 形式で出力する。

| 属性名 | 概要 |
| --- | --- |
| name | 名称 | 
| image_url | 代表画像のURL |
| facilities | 飼育園館 |
| habitat | 生息地 |
| size | 体の大きさ |
| food | えさ |
| feature | 特徴 |

出力例：

```
[
    {
        "name": "アカカンガルー",
        "image_url": "https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg",
        "facilities": [
            "多摩動物公園"
        ],
        "habitat": "オーストラリア大陸のほぼ全域",
        "size": "おとなのオスで体長1.3〜1.6ｍ、尾長0.8〜1ｍ、体重約80kg",
        "food": "主食は草で、木の芽や葉なども食べます。動物園では、牧草やニンジン、りんご、パンなどをあたえています。",
        "feature": "カンガルーは有袋類（ゆうたいるい＝おなかにある袋で子どもを育てる仲間）の代表的な動物です。ひらけた草原に10〜12頭の群れをつくってくらしています。昼間は直射日光をさけて休息し、夜間に草などを食べます。とびはねるときは後あしをそろえてとび、尾を地面につけていますが、速度をだすときは尾を上にあげ地面につけません。短距離では時速50km近い速さで走ることができます。"
    },
```

出力ファイル名は./animal_data_set.json で固定である。

### 動物名と画像URLを抽出してテーブル化する

filter.py を使用する。
本スクリプトを実行し、./animal_data_set.json から動物の情報を読み込んで成形し、
動物名と画像URLを抽出してテーブル化する。

下記のようにしてスクリプトを実行する。

実行例：
```bash
$ python .\filter.py
--- menu ---
1. 動物園の一覧を表示する
2. 指定された動物園にいる動物の一覧を表示する
999. 終了する
input: 2
0: 上野動物園
1: 井の頭自然文化園
2: 多摩動物公園
3: 大島公園動物園
4: 葛西臨海水族園
5: 都立動物園では飼育していません
input: 2
アカカンガルー  https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg
アジアゾウ      https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg
...
アカカンガルー  https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg
フサオネズミカンガルー  https://www.tokyo-zoo.net/Encyclopedia/Species//Y.fusao_nezumi_kanga_04.jpg
レッサーパンダ  https://www.tokyo-zoo.net/Encyclopedia/Species//X.redpanda.jpg
計：76
animal_list_in_place[2].tsv に一覧が出力されました
--- menu ---
1. 動物園の一覧を表示する
2. 指定された動物園にいる動物の一覧を表示する
999. 終了する
input: 999
$ 
```

スクリプトを起動後、「指定された動物園にいる動物の一覧を表示する」を選択して
所望の動物園を選択すると、

animal_list_in_place[X].tsv

という名前のファイルが出力されるので、内容を確認しておく。
（Xは選択した動物園の通し番号。）
本ファイルには下記４列が含まれているが、この段階では学名とTAXON_IDの欄は空欄である。

- 日本語名
- 学名
- ImageURL
- TAXON_ID

出力例：
```tsv
日本語名	学名	ImageURL	TAXON_ID
アカカンガルー		https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg	
アジアゾウ		https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg	
アフリカゾウ		https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AFURIKAZOU_1006.jpg	
...
```

## 動物に対する学名の付与(手動でデータを加工する)

動物にはそれぞれ学名が定義されており、上述したanimal_list_in_place[X].tsv の「学名」列にそれぞれの動物に対する学名を追記する。
現状学名の追記作業は手作業であるが、生成AIを使うなどして学名を付与することも可能である。
(詳細な手順は省略)

更新前のファイル例：
```tsv
日本語名	学名	ImageURL	TAXON_ID
アカカンガルー		https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg	
アジアゾウ		https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg	
アフリカゾウ		https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AFURIKAZOU_1006.jpg	
...
```

更新後のファイル例：
```tsv
日本語名	学名	ImageURL    TAXON_ID
アカカンガルー	Osphranter rufus	https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg
アジアゾウ	Elephas maximus	https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg
アフリカゾウ	Loxodonta africana	https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AFURIKAZOU_1006.jpg

...
```

## TAXON_ID付与

name2taxon_id.py を使用する

```
python ./name2taxon_id.py (path_to_table_file)
```

path_to_table_file には前述したanimal_list_in_place[X].tsv を指定し、animal_list_in_place[X].tsv の学名の欄をあらかじめ埋めておくこと。

```
日本語名	学名	ImageURL    TAXON_ID
```

本スクリプトは上記のように属性が並んでいるファイル（1行目はタイトル）から
学名を読み込み、学名からTAXON_ID を求めて書き込む。
元のファイルにはTAXON_ID が書かれていなくてもよく、
ダミー値が書かれていてもよい。
ImageURL は要素自体が存在しなければならないが、
空文字でもよい。

出力ファイル名はanimal_list_in_place[X].taxon_id.tsv である。

入力ファイルの例：

```
日本語名	学名	ImageURL    TAXON_ID
アカカンガルー	Osphranter rufus	https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg
アジアゾウ	Elephas maximus	https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg
アフリカゾウ	Loxodonta africana	

```

実行例：
```
python .\name2taxon_id.py .\animal_list_in_tama.txt
```

出力ファイルの例：

```
日本語名	学名	ImageURL	TAXON_ID
アカカンガルー	Osphranter rufus	https://www.tokyo-zoo.net/Encyclopedia/Species//Y.AKAKANNGARU-_1001.jpg	1453439
アジアゾウ	Elephas maximus	https://www.tokyo-zoo.net/Encyclopedia/Species//X.AJIAZOU_1003.jpg	43697
アフリカゾウ	Loxodonta africana		43694
```

## iNaturalist からcc0, cc-byに絞って画像を取得する

inaturalist_downloader.py を使用する。

第一引数にanimal_list_in_place[2].taxon_id.tsv へのパスを指定し、スクリプトを実行すると、images/XXX/ ディレクトリ配下にXXXで示される学名の動物画像を取得する。
iNaturalist からデータを取得する際にTAXGON_ID を使用するため、
animal_list_in_place[2].taxon_id.tsv にはTAXGON_ID の値が含まれていなければならない。

実行例：

```bash
python .\inaturalist_downloader.py .\animal_list_in_place[2].taxon_id.tsv
```

## 独自の画像を学習データにする（必要に応じて）

### 動画から画像を一枚ずつ抽出する

movie2picture.py を使用する。

引数で指定されたパスの動画を読み込み、中央部分の最大正方形領域を抽出して256x256サイズに縮小し、ファイルに保存する

```
positional arguments:
  input                 入力動画パス

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   出力ディレクトリ, デフォルト値:frames
  --size SIZE           出力画像サイズ（正方形）, デフォルト値:256
  --format {png,jpg,jpeg}
                        出力画像形式, デフォルト値: png
```


## データクレンジング

square_image_selector.py を使用する。

```
python .\square_image_selector.py (対象画像の入ったディレクトリへのパス)
```

上記を実行すると対象画像が一つずつ表示されるので、３つのどれかを実施する。

| やりたいこと | 操作内容 |
| --- | --- |
| 画像をそのまま使用する | スペースキーを押す |
| 画像を破棄する | DELキーを押す |
| 画像の一部分のみを使用する | マウスドラッグで範囲を選択してEnterキーを押す |

