import pandas as pd
import matplotlib.pyplot as pd_plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

df = pd.read_csv('cantopop_corpus_final_583_yue.csv')

# --- Data Cleaning ---
def extract_year(date_val):
    if pd.isna(date_val): return np.nan
    date_str = str(date_val).strip()
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(dt): return dt.year
    except: pass
    match = re.search(r'\d{4}', date_str)
    return int(match.group()) if match else np.nan

df['Year_Cleaned'] = df['Released on'].apply(extract_year)
df['Lyrics_Len'] = df['Lyrics_YuE'].str.len()

# --- 1. Top 12 Artists (Plot 1) ---
plt.figure(figsize=(10, 6))
top_artists = df['Artist'].value_counts().head(12)
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
plt.title('Top 12 Artists in Dataset', fontsize=14)
plt.xlabel('Number of Songs', fontsize=12)
plt.ylabel('Artist', fontsize=12)
plt.tight_layout()
plt.savefig('plot_top_artists_en.png')
plt.show()

# --- 2. Top 10 Lyricists (Plot 2) ---
plt.figure(figsize=(10, 6))
top_lyricists = df['Lyricist'].value_counts().head(10)
sns.barplot(x=top_lyricists.values, y=top_lyricists.index, palette='plasma')
plt.title('Top 10 Lyricists by Song Count', fontsize=14)
plt.xlabel('Number of Songs', fontsize=12)
plt.ylabel('Lyricist', fontsize=12)
plt.tight_layout()
plt.savefig('plot_top_lyricists_en.png')
plt.show()

# --- 3. Song Distribution by Year (Plot 3) ---
plt.figure(figsize=(12, 6))
sns.histplot(df['Year_Cleaned'].dropna(), bins=30, kde=True, color='teal')
plt.title('Song Distribution by Release Year (1980-2026)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('plot_timeline_en.png')
plt.show()

# --- 4. Average Lyrics Length (Plot 4) ---
plt.figure(figsize=(10, 6))
selected_artists = ['張國榮 (Leslie Cheung)', '泳兒 (Vincy Chan)', '謝安琪 (Kay Tse)', '容祖兒 (Joey Yung)', '陳奕迅 (Eason Chan)', '衛蘭 (Janice Vidal)']
avg_len_df = df[df['Artist'].isin(selected_artists)].groupby('Artist')['Lyrics_Len'].mean().sort_values(ascending=False)
sns.barplot(x=avg_len_df.values, y=avg_len_df.index, palette='magma')
plt.title('Average Lyrics Length by Selected Artists', fontsize=14)
plt.xlabel('Average Character Count', fontsize=12)
plt.ylabel('Artist', fontsize=12)
plt.tight_layout()
plt.savefig('plot_lyrics_length_en.png')
plt.show()