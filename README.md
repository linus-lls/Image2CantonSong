# Image2CantonSong

> Image-to-Cantonese-song generation system with multimodal lyric generation, style prompt control, YuE music generation, and evaluation modules.
>
> 圖像到粵語歌曲生成系統，整合多模態歌詞生成、音樂風格 Prompt 控制、YuE 音樂生成與多維度評估模組。

## Overview / 項目概述

**Image2CantonSong** is an end-to-end multimodal generation project that converts an input image into Cantonese lyrics and then generates a song through a lyrics-to-music pipeline. The system is designed as a research prototype and demo platform for exploring image understanding, Cantonese lyric generation, retrieval-augmented generation, music prompt engineering, and automatic evaluation.

**Image2CantonSong** 是一個端到端多模態生成項目，目標是將輸入圖片轉換為粵語歌詞，並進一步通過歌詞到音樂生成流程產生歌曲。本系統主要作為研究原型與展示平台，用於探索圖像理解、粵語歌詞生成、檢索增強生成、音樂 Prompt 設計，以及自動化評估方法。

The current main demo is located in:

目前主要 Demo 位於：

```bash
canto_project_official_yue_bridge_demo_v2/
```

## Main Workflow / 主要流程

```text
Image Upload
    ↓
Multimodal Image Understanding
    ↓
Cantonese Lyrics and Genre Prompt Generation
    ↓
Manual Confirmation / Editing
    ↓
Evaluation
    ↓
YuE-based Song Generation
    ↓
Generated Song Output
```

中文流程：

```text
圖片上傳
    ↓
多模態圖像理解
    ↓
生成粵語歌詞與音樂風格 Prompt
    ↓
人工確認 / 修改
    ↓
自動化評估
    ↓
基於 YuE 的歌曲生成
    ↓
輸出歌曲結果
```

## Key Features / 主要功能

### 1. Image-to-Lyrics Generation / 圖像到歌詞生成

The system accepts an uploaded image and uses a multimodal language model to generate a Cantonese lyric draft and related metadata.

系統支援使用者上傳圖片，並通過多模態語言模型生成粵語歌詞草稿與相關生成資訊。

The generated lyrics are expected to reflect:

生成歌詞應盡量反映：

- Main visual objects / 圖片中的主要物件
- Scene and background / 場景與背景
- Mood and emotion / 氛圍與情緒
- Possible story or theme / 潛在故事與主題
- Hong Kong Cantonese lyric style / 香港粵語歌詞風格

### 2. Multimodal Model Support / 多模態模型支援

The current Streamlit demo supports image-capable Hugging Face models, including:

目前 Streamlit Demo 支援圖像理解型 Hugging Face 模型，包括：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `OpenGVLab/InternVL2-4B`

The app also provides options for model temperature, maximum new tokens, CPU execution, and Hugging Face token input.

應用界面同時提供 temperature、max new tokens、CPU 運行，以及 Hugging Face Token 輸入等設置。

### 3. Cantonese Lyric Generation / 粵語歌詞生成

The project focuses on Cantonese lyric generation instead of generic Mandarin or written Chinese output.

本項目重點是生成粵語歌詞，而不是一般普通話或書面中文輸出。

The lyric generation process emphasizes:

歌詞生成過程重點考慮：

- Traditional Chinese characters / 繁體中文
- Cantonese expressions / 粵語表達
- Hong Kong-style Cantopop wording / 香港流行曲式用詞
- Structured lyric sections / 結構化歌詞段落
- Image-related storytelling / 與圖片內容相關的敘事

### 4. Style Prompt Control / 音樂風格 Prompt 控制

The demo supports three genre prompt modes:

Demo 支援三種音樂風格 Prompt 模式：

1. **Preset**  
   Use a predefined Cantopop-style prompt.

   **預設模式**  
   使用預先設計好的粵語流行曲風格 Prompt。

2. **Select tags from list**  
   Select style tags from categories such as genre, instrument, mood, gender, and timbre. The mandatory `Cantonese` tag is automatically included.

   **標籤選擇模式**  
   從 genre、instrument、mood、gender、timbre 等類別中選擇風格標籤，並自動加入必要標籤 `Cantonese`。

3. **Generate by lyrics model**  
   Let the multimodal lyrics model generate the genre prompt directly.

   **模型生成模式**  
   由多模態歌詞模型直接生成音樂風格 Prompt。

### 5. Retrieval-Augmented Generation / 檢索增強生成

For `OpenGVLab/InternVL2-4B`, the demo supports optional RAG-based lyric generation. It retrieves similar Cantopop lyrics from the corpus and injects them as few-shot context.

對於 `OpenGVLab/InternVL2-4B`，Demo 支援可選的 RAG 歌詞生成模式。系統會從粵語流行歌語料中檢索相似歌詞，並作為 few-shot 參考加入生成 Prompt。

The default corpus file is:

默認語料文件為：

```bash
cantopop_corpus_final_583_yue.csv
```

### 6. Manual Review and Editing / 人工確認與修改

After lyric generation, users can manually edit:

歌詞生成後，使用者可以人工修改：

- Song title / 歌曲標題
- Lyrics / 歌詞
- Genre prompt / 音樂風格 Prompt

This design allows the system to combine automatic generation with human control before music generation.

此設計可以在音樂生成前結合自動生成與人工控制，減少格式錯誤或不合適內容對下游音樂生成的影響。

### 7. YuE Music Generation Bridge / YuE 音樂生成橋接

The final music generation stage uses the official YuE environment. The Streamlit demo works as a bridge: it prepares the lyric and prompt files, then calls the YuE inference script through an external Python environment by subprocess.

最終歌曲生成階段使用 YuE 官方環境。Streamlit Demo 作為橋接層，負責準備歌詞與 Prompt 文件，並通過 subprocess 調用外部 Python 環境中的 YuE 推理腳本。

The demo does **not** directly modify the original YuE source code.

Demo **不直接修改** YuE 原始碼，便於維持 YuE 官方環境的獨立性。

### 8. ICL Reference Audio / ICL 參考音頻

The demo keeps optional ICL settings:

Demo 保留可選 ICL 設置：

- Single-track ICL / 單軌 ICL
- Dual-track ICL / 雙軌 ICL
- Reference vocal and instrumental audio uploads / 參考人聲與伴奏音頻上傳
- Prompt start and end time control / Prompt 起止時間控制

### 9. Evaluation Modules / 評估模組

The demo integrates several evaluation tabs:

Demo 整合多個評估頁籤：

- Image-lyrics alignment / 圖像與歌詞語義一致性
- Image-lyrics emotion similarity / 圖像與歌詞情緒一致性
- Lyrics format evaluation / 歌詞格式評估
- Cantonese lyrics quality evaluation / 粵語歌詞質量評估

Additional scripts are also included for batch evaluation and model visualization.

項目中亦包含批量評估與模型視覺化相關腳本。

## Repository Structure / 倉庫結構

```text
Image2CantonSong/
├── .streamlit/                                  # Streamlit configuration
├── Dataset/                                     # Dataset resources
├── Emotion/                                     # Emotion-related models and scripts
├── Evaluation/                                  # Evaluation modules
│   ├── genre_alignment/                         # Genre prompt alignment evaluation
│   ├── image_lyrics_alignment/                  # Image-lyrics semantic alignment
│   ├── image_lyrics_emotion/                    # Image-lyrics emotion similarity
│   ├── lyrics_format/                           # Lyrics format evaluation
│   └── lyrics_quality/                          # Cantonese lyrics quality evaluation
│
├── Image-Prompt-Pairs/                          # Image-prompt pair resources
├── Images/                                      # Example images or image assets
├── Song_evaluation/                             # Song evaluation resources
├── YuE/                                         # YuE-related integration files
├── canto_project_official_yue_bridge_demo_v2/   # Main Streamlit demo
│   ├── app.py                                   # Main Streamlit application
│   ├── generator.py                             # Image-to-prompt and YuE generation logic
│   ├── schemas.py                               # Data schema definitions
│   ├── state_utils.py                           # Streamlit state utilities
│   ├── launch.sh                                # Launch script
│   ├── batch_eval_mm_models.py                  # Batch evaluation for multimodal models
│   ├── build_lyrics_review_excel_multi_outputs.py
│   ├── cli_image_to_prompt.py                   # CLI image-to-prompt generation
│   ├── visualize_model_radar.py                 # Radar chart visualization
│   ├── visualize_model_radar_subjective.py      # Subjective radar chart visualization
│   ├── examples/                                # Example prompt bundles
│   ├── modules/
│   │   ├── clean_yue_runtime.py                 # YuE runtime cleaning utilities
│   │   ├── mm_direct_gen.py                     # Multimodal lyric generation module
│   │   └── rag_retriever.py                     # RAG retrieval module
│   └── outputs/                                 # Generated outputs
│
├── envs/                                        # Conda environment specifications
├── cantopop_corpus_final_583_yue.csv            # Cantopop lyric corpus for RAG
├── finetune_internvl2_4b_cantopop.ipynb         # InternVL fine-tuning notebook
├── rag_internvl2_4b_cantopop.ipynb              # RAG experiment notebook
├── paths.py                                     # Project path configuration
└── README.md
```

## Environment Setup / 環境配置

This project uses multiple Conda environments because the Streamlit demo, multimodal generation, evaluation modules, and YuE generation may require different dependency versions.

本項目使用多個 Conda 環境，因為 Streamlit Demo、多模態生成、評估模組與 YuE 音樂生成可能需要不同版本的依賴。

### 1. Clone the repository / 下載倉庫

```bash
git clone https://github.com/Chr1sNga1/Image2CantonSong.git
cd Image2CantonSong
```

### 2. Create the main demo environment / 建立主要 Demo 環境

The main Streamlit and multimodal lyric generation environment is `yue_project_clean`.

主要 Streamlit 與多模態歌詞生成環境為 `yue_project_clean`。

```bash
cd envs
conda env create -f ./yue_project_clean/yue_project_clean.yml
conda activate yue_project_clean
```

If additional packages are required:

如仍有缺失依賴，可安裝：

```bash
pip install -r ./yue_project_clean/yue_project_clean-requirements.txt
```

If CUDA-specific PyTorch wheels are required, install the correct version according to your CUDA environment.

如需 CUDA 版本的 PyTorch，請根據本機 CUDA 版本安裝對應版本。

Example:

```bash
pip install -r ./yue_project_clean/yue_project_clean-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. Set up YuE environment / 配置 YuE 環境

The final song generation step depends on the official YuE environment. Please make sure YuE can run independently before connecting it with this demo.

最終歌曲生成部分依賴 YuE 官方環境。請先確認 YuE 推理腳本可以獨立運行，再將其與本 Demo 連接。

You may need to modify local paths for:

你可能需要根據自己的機器修改：

- YuE Python executable path / YuE Python 執行文件路徑
- YuE inference script path / YuE 推理腳本路徑
- Model paths / 模型路徑
- Output directory / 輸出目錄

### 4. Hugging Face token / Hugging Face Token

Some models may require Hugging Face access.

部分模型可能需要 Hugging Face 權限。

You can either log in from the command line:

可以通過命令行登入：

```bash
huggingface-cli login
```

or provide the token in the Streamlit sidebar.

或在 Streamlit 側邊欄輸入 Token。

For command-line usage:

命令行方式：

```bash
export HF_TOKEN=your_huggingface_token
```

## Running the Demo / 啟動 Demo

Activate the main environment:

啟動主要環境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project_clean
```

Go to the demo folder:

進入 Demo 目錄：

```bash
cd canto_project_official_yue_bridge_demo_v2
```

Install local requirements if needed:

如有需要，安裝 Demo 依賴：

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

啟動 Streamlit 應用：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

Debug mode:

Debug 模式：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 -- --debug
```

Presentation mode:

展示模式：

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 -- --presentation
```

If port `8501` is already in use:

如果 `8501` 端口已被佔用：

```bash
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

## Usage Guide / 使用方法

### English

1. Open the Streamlit interface.
2. Enter a Hugging Face token if required.
3. Upload an image.
4. Select a multimodal model.
5. Choose whether to use RAG if `InternVL2-4B` is selected.
6. Choose a genre prompt source:
   - Preset
   - Select tags from list
   - Generate by lyrics model
7. Select lyric length: 4, 8, or 16 lines.
8. Adjust model parameters if needed.
9. Click **Generate Lyrics & Prompt**.
10. Review and edit the generated title, lyrics, and genre prompt.
11. Run evaluation modules if needed.
12. Configure YuE generation settings.
13. Generate the final song output.

### 中文

1. 打開 Streamlit 界面。
2. 如模型需要權限，輸入 Hugging Face Token。
3. 上傳圖片。
4. 選擇多模態模型。
5. 如果選擇 `InternVL2-4B`，可選擇是否啟用 RAG。
6. 選擇音樂風格 Prompt 來源：
   - Preset
   - Select tags from list
   - Generate by lyrics model
7. 選擇歌詞長度：4、8 或 16 行。
8. 如有需要，調整模型生成參數。
9. 點擊 **Generate Lyrics & Prompt**。
10. 檢查並修改生成的標題、歌詞與音樂風格 Prompt。
11. 如有需要，運行評估模組。
12. 配置 YuE 生成參數。
13. 生成最終歌曲輸出。

## Lyric Format / 歌詞格式

The project expects structured Cantonese lyrics. Section tags should be placed on separate lines.

本項目要求生成結構化粵語歌詞，段落標籤應單獨成行。

Example:

```text
[verse]
望住霓虹慢慢散
街角風聲又再返

[chorus]
如果今晚仲有夢
可唔可以留低一分鐘

[end]
```

Recommended rules:

建議規則：

- Use Traditional Chinese. / 使用繁體中文。
- Use Cantonese expressions. / 使用粵語表達。
- Keep section tags lowercase. / 段落標籤保持小寫。
- Put `[end]` as the final non-empty line. / `[end]` 應為最後一個非空行。
- Avoid explanations, bullet points, or Markdown inside the lyrics. / 歌詞中避免加入解釋、項目符號或 Markdown。
- Keep the lyrics related to the uploaded image. / 歌詞內容應與上傳圖片相關。

Common tags:

常見標籤：

```text
[verse]
[chorus]
[bridge]
[outro]
[end]
```

## Evaluation / 評估

### 1. Image-Lyrics Alignment / 圖像與歌詞語義一致性

Evaluates whether the generated lyrics are semantically related to the input image, using CLIP-style image-text similarity.

評估生成歌詞是否與輸入圖片在語義上相關，可使用類似 CLIP 的圖文相似度方法。

### 2. Image-Lyrics Emotion Similarity / 圖像與歌詞情緒一致性

Compares the emotional cues from the image or user-selected mood tags with the emotions expressed in the generated lyrics.

比較圖片情緒或使用者選擇的 mood tags 與生成歌詞中的情緒是否一致。

### 3. Lyrics Format Evaluation / 歌詞格式評估

Checks section tags, blank lines, structural similarity, and unwanted extra text.

檢查段落標籤、空行、結構相似度，以及是否存在多餘解釋文字。

### 4. Cantonese Lyrics Quality Evaluation / 粵語歌詞質量評估

Evaluates Cantonese naturalness, lyric fluency, rhyme-related features, and overall lyric quality.

評估粵語自然度、歌詞流暢度、押韻相關特徵，以及整體歌詞質量。

## Utility Scripts / 工具腳本

The demo folder contains several utility scripts:

Demo 目錄包含多個輔助腳本：

| Script | Description | 中文說明 |
|---|---|---|
| `app.py` | Main Streamlit application | 主要 Streamlit 應用 |
| `generator.py` | Image-to-prompt and YuE generation logic | 圖像到 Prompt 及 YuE 生成邏輯 |
| `cli_image_to_prompt.py` | Command-line image-to-prompt generation | 命令行圖像到 Prompt 生成 |
| `batch_eval_mm_models.py` | Batch evaluation for multimodal models | 多模態模型批量評估 |
| `build_lyrics_review_excel_multi_outputs.py` | Build lyric review Excel files | 生成歌詞人工評審 Excel |
| `visualize_model_radar.py` | Radar chart visualization | 雷達圖視覺化 |
| `visualize_model_radar_subjective.py` | Subjective radar chart visualization | 主觀評估雷達圖視覺化 |

## Technologies Used / 使用技術

| Category | Technologies |
|---|---|
| Web Demo | Streamlit |
| Programming | Python |
| Multimodal Models | Qwen2.5-VL, InternVL |
| Deep Learning | PyTorch |
| Model Platform | Hugging Face Transformers |
| Retrieval | Retrieval-Augmented Generation |
| Music Generation | YuE |
| Evaluation | CLIP-style image-text similarity, emotion similarity, lyric format scoring |
| Data Processing | pandas, JSON, CSV |
| Visualization | Radar charts |
| Environment Management | Conda |

## Troubleshooting / 常見問題

### 1. GitHub README is not rendered / GitHub README 沒有渲染

If the root file is named `README` without `.md`, GitHub may display the raw Markdown text instead of rendering it.

如果根目錄文件名是沒有 `.md` 後綴的 `README`，GitHub 可能會直接顯示 Markdown 原文，而不是渲染格式。

Recommended fix:

建議修改為：

```bash
git mv README README.md
git add README.md
git commit -m "Rename README to README.md"
git push origin main
```

If `README.md` already exists:

如果已存在 `README.md`：

```bash
git rm README
git add README.md
git commit -m "Update README"
git push origin main
```

### 2. Port already in use / 端口被佔用

```bash
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

### 3. CUDA is not available / 無法使用 CUDA

Check GPU availability:

檢查 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `False`, check NVIDIA driver, CUDA version, PyTorch version, and Conda environment.

如果輸出為 `False`，請檢查 NVIDIA 驅動、CUDA 版本、PyTorch 版本與 Conda 環境。

### 4. Hugging Face token error / Hugging Face Token 錯誤

```bash
export HF_TOKEN=your_huggingface_token
```

or use:

或使用：

```bash
huggingface-cli login
```

### 5. YuE generation does not produce expected vocals / YuE 未生成預期人聲

Possible reasons:

可能原因包括：

- Incorrect lyric format / 歌詞格式不正確
- Inappropriate genre prompt / 音樂風格 Prompt 不合適
- Insufficient generation length / 生成長度不足
- YuE environment configuration issue / YuE 環境配置問題
- GPU memory limitation / GPU 記憶體不足

Try simplifying the lyrics and genre prompt first.

建議先簡化歌詞與音樂風格 Prompt。

## Known Limitations / 已知限制

- Cantonese singing quality depends heavily on the downstream music generation model.  
  粵語演唱質量高度依賴下游音樂生成模型。

- Generated lyrics may still require manual editing.  
  生成歌詞仍可能需要人工修改。

- Long lyrics and long music generation can be unstable under limited GPU memory.  
  在 GPU 記憶體有限的情況下，長歌詞與長音樂生成可能不穩定。

- Some models require Hugging Face access permissions or large local storage.  
  部分模型需要 Hugging Face 權限或較大的本地存儲空間。

- Evaluation scores should be treated as reference indicators rather than absolute judgments.  
  評估分數應作為參考指標，而不是絕對判斷。

## Future Improvements / 未來改進方向

- Improve Cantonese pronunciation in generated songs.  
  提升生成歌曲中的粵語發音質量。

- Improve lyric-to-melody alignment.  
  改進歌詞與旋律匹配度。

- Expand the Cantopop lyric corpus for better RAG retrieval.  
  擴大粵語流行歌語料，提高 RAG 檢索質量。

- Add automatic lyric format repair.  
  增加自動歌詞格式修復功能。

- Improve long-song generation stability.  
  提升長歌曲生成穩定性。

- Add more complete human evaluation and quantitative evaluation.  
  增加更完整的人工評估與量化評估。

- Provide a cleaner deployment configuration.  
  提供更清晰的部署配置。

## Acknowledgement / 致謝

This project builds upon open-source tools and models from the multimodal generation, language modeling, retrieval, evaluation, and music generation communities.

本項目基於多模態生成、大語言模型、檢索增強生成、評估方法與音樂生成社群中的開源工具和模型構建。

Related tools and frameworks include:

相關工具與框架包括：

- Streamlit
- PyTorch
- Hugging Face Transformers
- Qwen2.5-VL
- InternVL
- YuE
- CLIP-style image-text similarity methods
- Retrieval-Augmented Generation methods

## License and Usage / 授權與使用說明

This repository is mainly intended for academic research, coursework, prototyping, and demonstration purposes.

本倉庫主要用於學術研究、課程項目、原型開發與展示用途。

Please check the licenses of all third-party models, datasets, and dependencies before using this project for commercial or public deployment.

如需將本項目用於商業用途或公開部署，請先檢查所有第三方模型、數據集與依賴庫的授權條款。

## Project Status / 項目狀態

The current version implements the main image-to-Cantonese-song pipeline, including image upload, multimodal lyric generation, prompt editing, YuE bridge generation, and several evaluation modules.

目前版本已實現主要圖像到粵語歌曲生成流程，包括圖片上傳、多模態歌詞生成、Prompt 修改、YuE 橋接生成，以及多個評估模組。

The project is still a research prototype and may require environment-specific path configuration before running on a new machine.

本項目仍屬於研究原型，在新機器上運行前可能需要根據環境修改本地路徑配置。
