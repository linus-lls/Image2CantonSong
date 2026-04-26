# Canto Project Demo — Official YuE Bridge (v2)

這個 demo 保留原本流程：

1. 上傳圖片
2. 用多模態 LLM 生成歌詞與 prompt 草稿
3. 人工確認 / 編輯
4. 最後一步用原版 YuE 官方環境生成歌曲

## 核心設計
- Streamlit / 多模態 LLM 跑在 `yue_project_clean`
- 最後一步不用當前環境執行 YuE
- 而是用 subprocess 直接調用：

`/userhome/cs5/u3665806/anaconda3/envs/yue_official/bin/python`

去執行：

`/userhome/cs5/u3665806/YuE/inference/infer.py`

## v2 修復
- 修復 `st.image(..., width="stretch")` 與 Streamlit 1.40.1 不兼容的問題
- 改成 `use_container_width=True`
- 多模態模組改為延遲導入 torch，避免 app 啟動階段直接炸掉
- 保留 single-track ICL / dual-track ICL

## 啟動
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project_clean
cd ~/canto_project_official_yue_bridge_demo_v2
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 127.0.0.1

# For debug mode. E.g. to load example lyrics to skip running image -> lyrics
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 -- --debug
```

## 注意
這份 demo 不會修改 YuE 源碼。
