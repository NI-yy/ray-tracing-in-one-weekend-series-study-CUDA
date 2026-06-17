# CUDA 化準備メモ

このメモは、実際に CUDA でレンダラを書き始める前に行った準備作業をまとめたものです。

## 開始時点

- 元の CPU 版レイトレーサから、CUDA 学習用の別リポジトリを作成した。
- CUDA 側のリポジトリは、元リポジトリの「最初のレンダリング画像が出た段階」のコミットに戻した。
- 最終版のコードは複雑なので、CUDA 化しやすい小さな状態から始める方針にした。

## CPU 版の基準値

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

## CUDA Toolkit

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

## Visual Studio の C++ コンパイラ

Windows で CUDA を使う場合、CUDA Toolkit だけでなく Microsoft C++ コンパイラの `cl.exe` も必要になる。

通常の PowerShell では `cl` が見えなかったが、Visual Studio の開発環境を読み込むと使えることを確認した。

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cl && nvcc --version'
```

確認できた環境:

- Visual Studio 2026 Community
- MSVC 19.51
- CUDA 13.3

## CMake の変更

`CMakeLists.txt` を更新し、CUDA を optional にした。

- `USE_CUDA` オプションを追加した。
- CMake が CUDA compiler の有無を確認するようにした。
- CUDA が見つかった場合だけ CUDA language support を有効化する。
- CUDA が見つからない環境でも CPU 版としてビルドできるようにした。

これにより、CUDA がない PC でもプロジェクト自体は壊れずに使える。

## NMake での CUDA ビルド

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

## 現在の状態

- CPU 版レンダラは引き続き動作する。
- Visual Studio 開発環境を読み込めば、CMake から CUDA compiler を検出できる。
- まだ CUDA でのレンダリング処理は実装していない。
- 次の作業は、最小の `.cu` ファイルを追加して、GPU 側で単純なグラデーション画像を生成すること。
