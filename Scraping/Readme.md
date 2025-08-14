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

取得したHTMLデータはcache ディレクトリ配下に格納される。
