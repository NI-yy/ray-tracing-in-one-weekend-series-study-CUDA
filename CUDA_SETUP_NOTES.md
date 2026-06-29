# CUDA 化作業メモ

このメモは、CPU 版レイトレーサを CUDA 化していくために、ここまで行った準備と実装を振り返るためのものです。

## マイルストーン

CUDA 化は、CPU 版の完成形を一気に移植せず、小さい処理から順に GPU 側へ移していく方針。

- [x] CUDA Toolkit と `nvcc` を使えるようにする。
- [x] Visual Studio の `cl.exe` と CUDA を CMake から検出できるようにする。
- [x] `src/cuda_renderer.cu` / `src/cuda_renderer.h` を追加し、C++ の `main.cpp` から CUDA 側の関数を呼べるようにする。
- [x] GPU で単純な背景グラデーションを生成する。
- [x] 球 1 個との交差判定を GPU で行う。
- [x] 複数の球を GPU 側で扱えるようにする。
- [x] GPU 側に簡単なマテリアル処理を追加する。
- [x] ランダムサンプリングと複数 bounce を GPU 側に移す。
- [x] CPU 版と CUDA 版を同じ条件で比較する。
- [x] CUDA 側に metal マテリアルを追加する。
- [x] CUDA 側に dielectric マテリアルを追加する。

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
bool render_cuda_single_sphere(const char* output_path, int image_width, int image_height);
bool render_cuda_multiple_spheres(const char* output_path, int image_width, int image_height);
```

`main.cpp` では、CUDA が有効な場合だけこの関数を呼ぶようにした。

```cpp
#ifdef RTWEEKEND_CUDA_ENABLED
    render_cuda_multiple_spheres("image_cuda_spheres.ppm", 200, 112);
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

## 9. GPU で球 1 個との交差判定を行う

背景グラデーションの次の段階として、CUDA kernel 内で ray を作り、球 1 個との交差判定を行う処理を追加した。

この段階では、CPU 版の `sphere` / `hittable` / `material` / `shared_ptr` は使わず、CUDA 側だけで使う小さな構造体を `cuda_renderer.cu` に用意した。

- `cuda_vec3`
- `cuda_ray`
- `cuda_sphere`

処理の流れ:

1. 各ピクセルを 1 CUDA thread が担当する。
2. ピクセル位置から camera ray を作る。
3. `hit_sphere()` で ray と球の交差判定を行う。
4. hit した場合は法線カラーで描画する。
5. miss した場合は背景グラデーションを描画する。

出力先:

```text
image_cuda_sphere.ppm
```

CUDA 実行ログ:

```text
CUDA single sphere render:
  output: image_cuda_sphere.ppm
  image: 200x112
  time: 0.30904 seconds
```

この時点で、GPU 側でレイトレーシングの最小形である「ray を飛ばして球に当たるか調べる」処理まで到達した。

## 10. 複数の球を GPU 側で扱う

球 1 個の交差判定が動いたので、次に複数の球を GPU 側で扱えるようにした。

CPU 側で `cuda_sphere` の配列を作成し、`cudaMemcpy` で GPU メモリへ転送する。

現在の確認用シーンでは、次の 4 個の球を使っている。

- 地面用の巨大な球
- 中央の球
- 左の球
- 右の球

GPU kernel 側では、各ピクセルごとに全ての球を順番に調べ、一番近い交差を採用する。

処理の流れ:

1. CPU 側で `std::vector<cuda_sphere>` を作る。
2. `cudaMalloc` で GPU 側の sphere 配列を確保する。
3. `cudaMemcpyHostToDevice` で sphere 配列を GPU に送る。
4. 各 CUDA thread が ray を作る。
5. `hit_sphere_list()` で全 sphere をループし、一番近い hit を探す。
6. hit した場合は法線カラーで描画する。
7. miss した場合は背景グラデーションを描画する。

出力先:

```text
image_cuda_spheres.ppm
```

CUDA 実行ログ:

```text
CUDA multiple spheres render:
  output: image_cuda_spheres.ppm
  image: 200x112
  spheres: 4
  time: 0.275578 seconds
```

この段階では、まだマテリアルごとの色や反射・散乱は実装していない。全ての球は法線カラーで表示している。

## 11. GPU 側に簡単なマテリアル処理を追加する

複数球を扱えるようになったので、次に球ごとの固定色を GPU 側で扱えるようにした。

CPU 版の本格的な `material` 階層はまだ移植せず、まずは `cuda_sphere` に `albedo` を直接持たせる形にした。

```cpp
struct cuda_sphere {
    cuda_vec3 center;
    double radius;
    cuda_vec3 albedo;
};
```

また、hit した球の情報を保持する `hit_record` にも `albedo` を追加した。

```cpp
struct hit_record {
    double t;
    cuda_vec3 point;
    cuda_vec3 normal;
    cuda_vec3 albedo;
};
```

これにより、どの球に hit したかに応じて、球ごとの色を使って描画できるようになった。

ただし、固定色をそのまま出すだけだと球の立体感が分かりにくい。そのため、簡単な光方向を 1 つ決めて、法線との dot で明るさを変える簡易 Lambert 風の shading を入れた。

```cpp
const cuda_vec3 light_direction = unit_vector(make_vec3(-1.0, 1.0, 0.5));
const double diffuse = fmax(0.0, dot(rec.normal, light_direction));
const double ambient = 0.25;
return (ambient + 0.75 * diffuse) * rec.albedo;
```

これはまだ本格的な物理ベースマテリアルではない。

- 反射はしない。
- 屈折はしない。
- ランダム散乱はしない。
- bounce もしない。

目的は、GPU 側で「物体ごとに見た目の情報を持つ」ための最初の一歩。

## 12. ランダムサンプリングと複数 bounce を GPU 側に移す

簡単なマテリアル色と直接照明風の shading までできたので、次に Ray Tracing in One Weekend の本質に近い処理である、ランダムサンプリングと複数 bounce を CUDA 側へ移した。

今回の実装は、コピー元リポジトリの最終コミットにある本格的な `lambertian` / `metal` / `dielectric` の完全移植ではなく、まず Lambertian 風の diffuse bounce に絞った第一段階。

追加した主な処理:

- GPU 側で使う簡易乱数 `xorshift32`
- 1 ピクセル内で少しずつ位置をずらすランダムサンプリング
- `random_in_unit_sphere()` / `random_unit_vector()` によるランダム散乱方向
- 再帰ではなく `for` ループによる複数 bounce
- hit した球の `albedo` を attenuation として掛けていく処理
- サンプル平均後の簡単なガンマ補正

CPU 版の `ray_color(r, depth, world)` は再帰で書かれているが、CUDA では再帰を避け、次のような考え方でループにした。

```cpp
cuda_vec3 attenuation = make_vec3(1.0, 1.0, 1.0);

for (int depth = 0; depth < max_depth; depth++) {
    if (!hit_sphere_list(...))
        return attenuation * background_color(ray);

    scatter_direction = rec.normal + random_unit_vector(rng_state);
    attenuation = attenuation * rec.albedo;
    ray = cuda_ray{rec.point, scatter_direction};
}

return make_vec3(0.0, 0.0, 0.0);
```

`main.cpp` からは次の関数を呼ぶようにした。

```cpp
render_cuda_path_traced_spheres("image_cuda_path_traced.ppm", 200, 112, 20, 10);
```

この設定では以下の意味になる。

- 画像サイズ: 200 x 112
- samples per pixel: 20
- max depth: 10
- total primary samples: 448,000

実行ログ:

```text
CUDA path traced spheres render:
  output: image_cuda_path_traced.ppm
  image: 200x112
  spheres: 4
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  time: 0.289833 seconds
  primary samples/sec: 1.54572e+06
```

この段階で、GPU 側でも「レイが当たる、散乱する、次のレイを追跡する」という path tracing の基本形まで到達した。

まだ実装していないもの:

- metal の鏡面反射
- dielectric の屈折
- defocus blur
- コピー元の最終シーンと同じ大量のランダム球
- CPU 版と CUDA 版を完全に同じ条件にした公平な速度比較

## 13. CPU 版と CUDA 版を同じ条件で比較する

CUDA 側でランダムサンプリングと複数 bounce が動くようになったので、CPU 版と CUDA 版の条件をできるだけそろえて計測した。

比較のため、CPU 側のシーンを CUDA 側と同じ内容に変更した。

- 地面用の大きな球: `center = (0, -100.5, -1)`, `radius = 100.0`
- 中央の球: `center = (0, 0, -1)`, `radius = 0.5`
- 左の球: `center = (-1.0, 0, -1.2)`, `radius = 0.45`
- 右の球: `center = (1.0, 0, -1.2)`, `radius = 0.45`
- すべて Lambertian 風の diffuse bounce

カメラとレンダー設定もそろえた。

- 画像サイズ: 200 x 112
- samples per pixel: 20
- max depth: 10
- lookfrom: `(0, 0, 0)`
- lookat: `(0, 0, -1)`
- vfov: 90
- defocus blur: なし

CPU 側の出力先を分けるため、`camera::render()` に出力ファイル名を渡せるようにした。

```cpp
void render(const hittable& world, const char* output_path = "image.ppm")
```

今回の出力:

```text
CUDA: image_cuda_path_traced.ppm
CPU : image_cpu_comparison.ppm
```

計測結果:

```text
CUDA path traced spheres render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  time: 0.24982 seconds
  primary samples/sec: 1.79329e+06

CPU render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  Render time: 1.09046 seconds
  Primary samples/sec: 410835
```

この条件では、CUDA 版の方が CPU 版より約 4.4 倍速かった。

ただし、これはまだ厳密なベンチマークではない。

- CUDA 側はメモリ確保、GPU 実行、CPU へのコピー、PPM 書き出しを含む。
- CPU 側は PPM 書き出しを含む。
- 乱数の実装は CPU と CUDA で異なる。
- CPU 版は仮想関数と `shared_ptr` ベースの既存構造を使っている。
- CUDA 版は比較用の単純な構造体とループで実装している。

それでも、同じ解像度・同じサンプル数・同じ bounce 回数・同じシーンで比較できる状態にはなった。

## 14. CUDA 側に metal マテリアルを追加する

Lambertian 風の diffuse bounce だけでは、すべての球がざらっとした拡散反射になる。
コピー元の CPU 版に近づけるため、次の段階として CUDA 側に metal マテリアルを追加した。

今回の方針:

- `cuda_sphere` に `material_type` と `fuzz` を追加する。
- `hit_record` に hit した球のマテリアル情報も保存する。
- `material_lambertian` と `material_metal` を整数で区別する。
- metal の場合は、入射レイを法線で反射させる。
- `fuzz` を使って反射方向に少しランダム性を加える。

CUDA 側では、CPU 版のように `material` 基底クラスと仮想関数を使わず、単純な分岐で scatter 処理を切り替える形にした。

```cpp
if (rec.material_type == material_metal) {
    const cuda_vec3 reflected = reflect(unit_vector(ray.direction), rec.normal);
    const cuda_vec3 scatter_direction = reflected + rec.fuzz * random_unit_vector(rng_state);
    scattered = cuda_ray{rec.point, scatter_direction};
    did_scatter = dot(scattered.direction, rec.normal) > 0.0;
} else {
    cuda_vec3 scatter_direction = rec.normal + random_unit_vector(rng_state);
    scattered = cuda_ray{rec.point, scatter_direction};
}
```

比較しやすいように、CPU 側の小さな比較シーンも同じ構成にした。

- 地面: Lambertian
- 中央の球: Lambertian
- 左の球: Metal, `fuzz = 0.15`
- 右の球: Metal, `fuzz = 0.05`

実行結果:

```text
CUDA path traced spheres render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  time: 0.283181 seconds
  primary samples/sec: 1.58203e+06

CPU render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  Render time: 1.10968 seconds
  Primary samples/sec: 403720
```

この段階で、CUDA 側でも拡散反射だけでなく、鏡面反射に近い見た目を扱えるようになった。

## 15. CUDA 側に dielectric マテリアルを追加する

metal の反射が動いたので、次にガラスのような屈折を行う dielectric マテリアルを CUDA 側へ追加した。

CPU 版の `dielectric::scatter()` では、レイが物体の外から入る場合と内側から出る場合で屈折率の比を切り替える。
そのため CUDA 側でも、hit 時に `front_face` を記録するようにした。

追加した主な情報:

- `cuda_sphere::refraction_index`
- `hit_record::refraction_index`
- `hit_record::front_face`
- `material_dielectric`

追加した主な関数:

- `refract()`
- `reflectance()`

dielectric の scatter は、次のような流れにした。

```cpp
const cuda_vec3 unit_direction = unit_vector(ray.direction);
const double ri = rec.front_face ? (1.0 / rec.refraction_index) : rec.refraction_index;
const double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
const bool cannot_refract = ri * sin_theta > 1.0;

if (cannot_refract || reflectance(cos_theta, ri) > random_double(rng_state))
    direction = reflect(unit_direction, rec.normal);
else
    direction = refract(unit_direction, rec.normal, ri);
```

比較用の小さなシーンは次の構成にした。

- 地面: Lambertian
- 中央の球: Dielectric, `refraction_index = 1.5`
- 左の球: Metal, `fuzz = 0.15`
- 右の球: Metal, `fuzz = 0.05`

実行結果:

```text
CUDA path traced spheres render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  time: 0.245318 seconds
  primary samples/sec: 1.8262e+06

CPU render:
  image: 200x112
  samples_per_pixel: 20
  max_depth: 10
  total primary samples: 448000
  Render time: 1.05006 seconds
  Primary samples/sec: 426644
```

この段階で、CUDA 側でも Lambertian / Metal / Dielectric の 3 種類の基本マテリアルを扱えるようになった。

## 16. 途中で出た問題

最初の実行では、CUDA kernel 起動時に次のエラーが出た。

```text
CUDA error during gradient_kernel launch: the provided PTX was compiled with an unsupported toolchain.
```

原因は、CMake が CUDA architecture を `75` として構成していたこと。使用している GPU は RTX 4070 系なので、`89-real` を使うようにして解決した。

また、CUDA のヘッダ由来で `C4819` の文字コード警告が出るが、現時点ではビルドと実行は成功している。

## 17. 現在の状態

- CPU 版レンダラは引き続き動作する。
- CUDA compiler は Visual Studio 開発環境経由で検出できる。
- C++ の `main.cpp` から CUDA 側の関数を呼べる。
- GPU 側で単純な背景グラデーション画像を生成できる。
- GPU 側で球 1 個との交差判定を行い、法線カラーで描画できる。
- GPU 側で複数の球を配列として扱い、一番近い交差を選べる。
- GPU 側で球ごとの固定色 `albedo` を扱い、簡単な陰影を付けて描画できる。
- GPU 側でランダムサンプリングと複数 bounce を行い、Lambertian 風の簡易 path tracing ができる。
- CPU と CUDA を同じ小さなシーン・同じレンダー設定で比較できる。
- CUDA 側で metal マテリアルの反射と fuzz を扱える。
- CUDA 側で dielectric マテリアルの反射・屈折を扱える。

## 次にやること

コピー元の最終コミットに近づけるため、次はマテリアル、カメラ、シーン生成を順番に CUDA 側へ移していく。

今の CUDA 版は Lambertian 風の diffuse bounce までなので、次のマイルストーンは以下の通り。

- [x] CUDA 側に metal マテリアルを追加する。
- [x] CUDA 側に dielectric マテリアルを追加する。
- [ ] CUDA 側のカメラ設定を CPU 版に近づける。
- [ ] defocus blur を CUDA 側で扱う。
- [ ] ランダムな多数の球を CUDA 側で扱えるようにする。
- [ ] コピー元の最終シーンに近い構成で CPU 版と CUDA 版を比較する。
