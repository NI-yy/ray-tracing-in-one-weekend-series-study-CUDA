# ray-tracing-in-one-weekend-study-CUDA

Ray Tracing in One Weekend 系列を読みながら実装した CPU 版レイトレーサを、CUDA で少しずつ GPU 化していくための学習用リポジトリです。

コピー元の完成形を一気に CUDA 化するのではなく、次のように小さい処理から順番に GPU 側へ移しています。

- CUDA Toolkit / `nvcc` を使ったビルド環境の準備
- C++ の `main.cpp` から CUDA 側の関数を呼び出す構成
- GPU で背景グラデーションを生成
- GPU で球 1 個との交差判定
- GPU で複数の球との交差判定
- GPU 側で簡単なマテリアル色を扱う
- GPU 側でランダムサンプリングと複数 bounce を処理
- CPU 版と CUDA 版を同じ条件にして速度比較

現在の CUDA 実装では、複数の球に対して Lambertian / Metal / Dielectric の簡易マテリアルを扱い、複数サンプル・複数 bounce の簡易パストレーシングを GPU 側で実行しています。

## 現在の確認結果

CPU 版と CUDA 版を、次の条件にそろえて比較しました。

- 画像サイズ: 200 x 112
- Samples per pixel: 5
- Max depth: 10
- Total primary samples: 112,000
- シーン: コピー元の最終シーンに近い構成
- 球数: 485
- マテリアル: Lambertian + Metal + Dielectric
- Camera: `lookfrom = (13, 2, 3)`, `vfov = 20`
- Defocus angle: 0.6

計測結果:

- CUDA render time: 約 0.30 秒
- CUDA primary samples/sec: 約 379,252
- CPU render time: 約 13.71 秒
- CPU primary samples/sec: 約 8,170

詳細な作業ログは `CUDA_SETUP_NOTES.md` にまとめています。
