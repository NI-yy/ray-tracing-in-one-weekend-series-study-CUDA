# CUDA 化作業メモ

このメモは、CPU 版レイトレーサを CUDA 化していくために、ここまで行った準備と実装を振り返るためのものです。

## 1. 開始時点

- 元の CPU 版レイトレーサから、CUDA 学習用の別リポジトリを作成した。
- CUDA 側のリポジトリは、元リポジトリの「最初のレンダリング画像が出た段階」のコミットに戻した。
- 最終版のコードは複雑なので、CUDA 化しやすい小さな状態から始める方針にした。

## 2. CPU 版の基準値

動作確認しやすいように、CPU 版のレンダリング設定を軽くした。

- 画像サイズ: 200 x 112
- サンプル数: 5 samples per pixel
- 最大反射回数: 10
- 合計 primary sample 数: 112,000

計測結果:

- レンダリング時間: 5.88546 秒
- Pixels/sec: 3,805.99
- Primary samples/sec: 19,029.9

計測処理は `camera::render()` に `std::chrono` を使って追加した。

## 3. CUDA Toolkit と nvcc

CUDA Toolkit 13.3 をインストールした。

`nvcc` の確認:

```powershell
nvcc --version
```

結果:

```text
Cuda compilation tools, release 13.3, V13.3.33
```

`nvcc` は NVIDIA CUDA Compiler のことで、`.cu` ファイルや GPU 上で動く device code をコンパイルするためのコンパイラ。

## 4. Visual Studio の C++ コンパイラ

Windows で CUDA を使う場合、CUDA Toolkit だけでなく Microsoft C++ コンパイラの `cl.exe` も必要になる。

通常の PowerShell では `cl` が見えなかったが、Visual Studio の開発環境を読み込むと使えることを確認した。

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cl && nvcc --version'
```

確認できた環境:

- Visual Studio 2026 Community
- MSVC 19.51
- CUDA 13.3

## 5. CMake の CUDA 対応

`CMakeLists.txt` を更新し、CUDA を optional にした。

- `USE_CUDA` オプションを追加した。
- CMake が CUDA compiler の有無を確認するようにした。
- CUDA が見つかった場合だけ CUDA language support を有効化する。
- CUDA が見つからない環境でも CPU 版としてビルドできるようにした。
- RTX 4070 系に合わせて、CUDA architecture は `89-real` を使うようにした。

これにより、CUDA がない PC でもプロジェクト自体は壊れず、CUDA がある環境では `.cu` ファイルもビルドできる。

## 6. NMake での CUDA ビルド

この環境の CMake には `Visual Studio 18 2026` ジェネレータがなかったため、Visual Studio 開発環境を読み込んだ上で `NMake Makefiles` を使った。

構成:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake -S . -B build_nmake_cuda -G "NMake Makefiles"'
```

ビルド:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build build_nmake_cuda'
```

確認結果:

```text
CUDA support enabled
[100%] Built target inOneWeekend
```

## 7. CUDA renderer の追加

CUDA 側の関数を C++ の `main.cpp` から呼べるか確認するために、次のファイルを追加した。

- `src/cuda_renderer.h`
- `src/cuda_renderer.cu`

追加した関数:

```cpp
bool render_cuda_gradient(const char* output_path, int image_width, int image_height);
```

`main.cpp` では、CUDA が有効な場合だけこの関数を呼ぶようにした。

```cpp
#ifdef RTWEEKEND_CUDA_ENABLED
    render_cuda_gradient("image_cuda.ppm", 200, 112);
#endif
```

これにより、CPU 版レンダリングを残したまま、GPU 側の簡単な処理を先に実行できる。

## 8. GPU で背景グラデーションを生成

最初の CUDA kernel として、1 ピクセルを 1 CUDA thread が担当する背景グラデーション生成を実装した。

出力先:

```text
image_cuda.ppm
```

生成する色は、レイトレーシング本の背景に近い、上が水色で下が白の縦グラデーション。

CUDA 実行ログ:

```text
CUDA gradient render:
  output: image_cuda.ppm
  image: 200x112
  time: 0.202083 seconds
```

この段階では、まだ球との交差判定やレイトレーシング処理は CUDA 側に移していない。

## 9. 途中で出た問題

最初の実行では、CUDA kernel 起動時に次のエラーが出た。

```text
CUDA error during gradient_kernel launch: the provided PTX was compiled with an unsupported toolchain.
```

原因は、CMake が CUDA architecture を `75` として構成していたこと。使用している GPU は RTX 4070 系なので、`89-real` を使うようにして解決した。

また、CUDA のヘッダ由来で `C4819` の文字コード警告が出るが、現時点ではビルドと実行は成功している。

## 10. 現在の状態

- CPU 版レンダラは引き続き動作する。
- CUDA compiler は Visual Studio 開発環境経由で検出できる。
- C++ の `main.cpp` から CUDA 側の関数を呼べる。
- GPU 側で単純な背景グラデーション画像を生成できる。
- CPU と CUDA の公平な速度比較はまだ行っていない。

## 次にやること

次は、GPU 側に「球 1 個との交差判定」を実装する。

そのために、まずは CPU 版の `hittable` / `material` / `shared_ptr` 構造をそのまま持ち込まず、CUDA 用の単純な `struct` から始める。
