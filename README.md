# ray-tracing-in-one-weekend-study-CUDA

Ray Tracing in One Weekend 系列を読みながら実装した CPU 版レイトレーサを、CUDA で少しずつ GPU 化していくための学習用リポジトリです。

コピー元の完成形を一気に CUDA 化するのではなく、次のように小さい処理から順番に GPU 側へ移しています。

- [x] CUDA Toolkit / `nvcc` を使ったビルド環境の準備
- [x] C++ の `main.cpp` から CUDA 側の関数を呼び出す構成
- [x] GPU で背景グラデーションを生成
- [x] GPU で球 1 個との交差判定
- [x] GPU で複数の球との交差判定
- [x] GPU 側で簡単なマテリアル色を扱う
- [x] GPU 側でランダムサンプリングと複数 bounce を処理
- [x] CPU 版と CUDA 版を同じ条件にして速度比較

現在の CUDA 実装では、複数の球に対して Lambertian 風のランダム散乱を行い、複数サンプル・複数 bounce の簡易パストレーシングを GPU 側で実行しています。

## 次のマイルストーン

コピー元の最終コミットに近づけるため、次はマテリアル、カメラ、シーン生成を順番に CUDA 側へ移していきます。

- [ ] CUDA 側に metal マテリアルを追加する
- [ ] CUDA 側に dielectric マテリアルを追加する
- [ ] CUDA 側のカメラ設定を CPU 版に近づける
- [ ] defocus blur を CUDA 側で扱う
- [ ] ランダムな多数の球を CUDA 側で扱えるようにする
- [ ] コピー元の最終シーンに近い構成で CPU 版と CUDA 版を比較する

## 現在の確認結果

CPU 版と CUDA 版を、次の条件にそろえて比較しました。

- 画像サイズ: 200 x 112
- Samples per pixel: 20
- Max depth: 10
- Total primary samples: 448,000
- シーン: 地面 + 3 個の球
- マテリアル: Lambertian 風の diffuse bounce

計測結果:

- CUDA render time: 約 0.25 秒
- CUDA primary samples/sec: 約 1,793,000
- CPU render time: 約 1.09 秒
- CPU primary samples/sec: 約 410,835

詳細な作業ログは `CUDA_SETUP_NOTES.md` にまとめています。
