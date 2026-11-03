# Delft3D 时间层序与含砂率交互查看器

在 Windows 桌面中选择一组 Delft3D `.dat/.def` 结果，按保存时刻及平面点选位置绘制 XZ、YZ 层序和含砂率剖面，并用 CIGVis 查看三维层面与剖面。每组源文件使用独立缓存，切换结果后重新读取网格和时间轴。

## 新电脑配置

推荐 Windows 10/11 64 位、Python 3.12、可用的 OpenGL 显卡驱动及可写的本地磁盘。内存和缓存空间取决于“保存时刻数 × 网格数”；较大项目建议至少 8GB 内存并在 SSD 上保留数 GB 空间。二维绘图使用 Matplotlib；三维绘图使用 CIGVis、VisPy 和 PySide6，需要目标桌面能够创建 OpenGL 上下文。[VisPy 安装文档](https://vispy.org/installation.html)说明了 GUI 后端要求。

1. 从 [Miniconda 官方安装页面](https://www.anaconda.com/docs/getting-started/miniconda/install)安装 Windows x86_64 版本，也可使用已有 Anaconda。无需在 PyCharm 中另装一套 Anaconda。
2. 从 GitHub 下载整个仓库 ZIP 并完整解压，或使用 `git clone`。不要只复制某个 `.py` 文件。建议保存到如 `D:\Delft3DViewer` 的可写目录。原始模拟结果可放在其他磁盘。
3. 打开 **Anaconda Prompt**，进入解压后的仓库根目录，再创建环境：

   ```bat
   cd /d D:\Delft3DViewer
   conda env create --file environment.yml
   conda activate delft3d
   python check_environment.py
   python check_environment.py --render
   ```

   环境安装需要联网，YAML 中不包含本机绝对安装路径。`conda env create --file` 是兼容现有 Conda 的创建方式，见 [Conda 环境管理文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。也可双击根目录 `install_environment.cmd`，该脚本创建同名环境并检查依赖。
4. 启动：

   ```bat
   conda activate delft3d
   python outputs\time_stratigraphy_preview\run_interactive.py
   ```

   以后可以双击根目录 **`start.cmd`**。如果 Conda 装在自定义位置，请从 Anaconda Prompt 运行启动脚本，或设置 `DELFT3D_PYTHON` 为该环境 `python.exe` 的完整路径。程序使用内置的只读纯 Python NEFIS 5 读取器，**不需要额外安装 NEFIS DLL，也不需要在新电脑上安装完整 Delft3D 来读取结果**。

## 首次打开数据与切换项目

1. 新电脑首次启动后会出现“打开 Delft3D 结果文件”对话框。在 `.dat`、`.def` 路径旁分别点击“浏览…”，会打开 Windows 文件选择窗口；选择结果数据 `.dat`，再确认其配套 `.def` 文件。
2. 点击“检查文件并读取泥沙组分”，确认砂、泥组分及其干床密度。旧项目中使用的砂 `1600 kg/m³`、泥 `500 kg/m³` 仅是该项目参数；新项目应依据对应 `.sed` 文件或模型设置填写。干床密度与颗粒密度不是同一个量。
3. 点击“读取并绘图”，等待校验和缓存构建。读取在后台线程中进行。首次导入大文件可能需要较长时间；再次打开同一组未变化的文件会复用缓存。不同源文件或组分设置使用独立缓存。
4. 在主窗口点击“打开其他 DAT / DEF 结果…”可再次选择另一组源文件。新结果的网格范围、时刻数量和初始选点会重新设置；“上部参考／中部参考／下部参考”按当前有效网格定位，不自动判定真实内扇、中扇、外扇。
5. 选择时刻、输入 X/Y 或点击平面图，再点击“更新剖面”。XZ 固定 Y 沿 X 展开，YZ 固定 X 沿 Y 展开；可同时绘制两个方向。坐标单位为米，横轴图示为千米，高程正向向上。
6. 右侧同时显示时间层序剖面和含砂率剖面。“二维含砂率”可切换分期净增量、估算补全及混合床平均。缺测补全使用同一阶段邻域，缺少邻域时采用该位置当期混合床平均值；补全位置和来源写入导出数据。
7. 点击“Cigvis三维展示”查看同一时刻的位置剖面；左键旋转、滚轮缩放、Shift＋左键平移。点击“导出当前结果”保存 PNG、NPZ、CSV 和参数 JSON。

关于色标、输出文件以及分期含砂率的含义，见 [详细交互使用说明](outputs/time_stratigraphy_preview/README_交互程序.md)。输出是原生网格上按最近单元提取的剖面，实际采用的固定坐标和偏差会显示在界面中。

导入缓存默认位于 `%LOCALAPPDATA%\Delft3DTools\Delft3DSectionViewer\source_cache`（以本机 Qt 返回的用户应用数据目录为准）。缓存中的路径和数据只保存在本机；Qt 的最近使用设置保存在当前用户配置中。导出默认位于 `outputs/time_stratigraphy_preview/exports`，可用入口参数 `--export-root` 指定其他可写目录。新电脑代码包不附带旧电脑缓存，必须自行选择模型文件。

## 在 PyCharm 中运行

1. 使用 **Open** 打开整个仓库目录。
2. 在 **Settings → Python Interpreter → Add Interpreter** 中选择现有本地 Conda 环境 `delft3d`，不要重新创建第二套环境。菜单名称可能随 PyCharm 版本变化，见 [PyCharm 解释器配置文档](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)。
3. 不确定解释器路径时，在已激活环境的 Anaconda Prompt 中执行 `python -c "import sys; print(sys.executable)"`，把输出路径填入 PyCharm。
4. 运行 `outputs/time_stratigraphy_preview/run_interactive.py`。工作目录建议为仓库根目录；入口脚本通过自身位置定位模块，无需修改源码中的磁盘路径。

## 环境版本与更新

本次验证使用 Windows x64、Python **3.12.16**、NumPy **2.5.4**、SciPy **1.18.1**、Matplotlib **3.11.2**、CIGVis **0.3.2**、VisPy **0.17.0**、PySide6 **6.12.0**、Pillow **13.0.0**。`environment.yml` 与 `requirements.txt` 固定这些直接依赖版本，间接依赖由安装工具求解；它们不是包含所有安装包的离线镜像。

已有同名环境时，可在确认代码版本后更新依赖：

```bat
conda env update --name delft3d --file environment.yml
conda activate delft3d
python check_environment.py --render
```

如果已有 Python 3.12 虚拟环境，也可执行 `python -m pip install -r requirements.txt`。使用 `python -m pip` 可确保安装到当前解释器。优先使用提供的版本；CIGVis 的三维 API 会随版本演进，直接升级全部依赖后应重新验证绘图与导出。[CIGVis 官方仓库](https://github.com/JintaoLee-Roger/cigvis)提供其安装与 API 文档。

## 支持范围与结果解释

当前读取器面向 **Delft3D NEFIS 5 分离式 `.dat/.def` map 输出**。网格须为以米为单位、与 X/Y 轴对齐的笛卡尔矩形网格，时间与泥沙输出应同步。当前层序与含砂率流程要求**恰好两个泥沙组分、一个保存的混合床层**，通过界面指定哪一个是砂、哪一个是泥；旋转/曲线网格、多组分/多层床存储、其他 NEFIS 版本、NetCDF 或普通文本 `.dat` 不会自动转换为兼容输入。程序遇到不支持的结构会给出错误信息。

层序由床面时间序列推断。分期含砂率由相邻保存时刻组分库存差估算，混合床平均值来自所选时刻实际库存，缺测补全是连续展示所用的估算。原模型只保存一个混合床层，因此这些图不能作为恢复真实层内岩性或证明分层砂量收支闭合的依据。

## 常见问题

| 现象 | 处理 |
|---|---|
| 双击后找不到 Conda | 从 Anaconda Prompt 启动，或设置 `DELFT3D_PYTHON`；解释器路径由 `python -c "import sys; print(sys.executable)"` 查询。 |
| `No module named ...` | 检查当前/PyCharm 解释器是否为 `delft3d`，重新执行环境检查。 |
| 三维白屏或 OpenGL 错误 | 在目标桌面运行 `python check_environment.py --render`；更新显卡驱动后重试。远程桌面或虚拟机的 OpenGL 支持取决于其配置。 |
| `.dat` 无法读取 | 确认是 NEFIS 5 map 数据，并选择匹配的 `.def`；普通文本 `.dat` 不属于此格式。 |
| 密度或库存闭合校验失败 | 根据模型 `.sed` 设置确认所选砂泥组分和干床密度，避免把颗粒密度当作干床密度。 |
| 模拟仍在写文件 | 优先选模拟已完成并关闭后的结果；输入在读取期间变化时校验会中止。 |
| 首次导入或补全较慢 | 初次需要建立床面与组分缓存；补全首次需要对当前时刻的完整网格计算，同一时刻换剖面会复用结果。 |

## 代码和数据分发

仓库只分发源代码、环境配置与文档。原始 `.dat/.def`、缓存、导出图、环境文件夹和用户配置已被 `.gitignore` 排除；请把模拟结果单独复制到新电脑后，通过界面选择读取。修改代码后上传前仍需检查待提交文件，避免把其他本地文件加入版本库。

随附 `outputs/delft3d_viewer/NOTICE.txt`、`LICENSE-LGPL-2.1.txt` 以及 [第三方许可说明](THIRD_PARTY_NOTICES.md)应与读取器一起保留。此仓库不包含 Delft3D 安装程序或授权文件。

维护者可执行 `python -m unittest discover -s outputs/time_stratigraphy_preview -p test_project_loader.py -v`，使用合成文件检查不同数据/密度配置的缓存隔离、失败回滚和缓存重用。GitHub Actions 在 Windows/Python 3.12 上运行这些测试及语法检查，不读取真实模型文件；目标电脑的 Qt/OpenGL 显示能力仍应通过 `check_environment.py --render` 单独验证。
