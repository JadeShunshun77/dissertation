import pandas as pd
import os

# 1. 动态获取当前项目的根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 修改读取路径：适配你刚刚保存的新文件名
file_mev = os.path.join(BASE_DIR, "raw_data_sources", "seed_mev_addresses.csv")
file_defi = os.path.join(BASE_DIR, "raw_data_sources", "dune_defi_addresses.csv")
file_cex = os.path.join(BASE_DIR, "raw_data_sources", "dune_cex_addresses.csv")

try:
    df_mev = pd.read_csv(file_mev)
    df_defi = pd.read_csv(file_defi)
    df_cex = pd.read_csv(file_cex)

    # 3. 合并数据并清洗
    df_final = pd.concat([df_mev, df_defi, df_cex], ignore_index=True)
    df_final['address'] = df_final['address'].astype(str).str.strip().str.lower()
    df_final = df_final.drop_duplicates(subset=['address'], keep='first')

    # 4. 保存终极种子文件
    output_name = os.path.join(BASE_DIR, "final_seed_addresses.csv")
    df_final.to_csv(output_name, index=False)

    print(f"🎉 合并成功！最终生成 {len(df_final)} 个种子节点。已保存为 {output_name}")
    print("-" * 20)
    print("你的节点类别分布如下：")
    print(df_final['label'].value_counts())

except FileNotFoundError as e:
    print(f"❌ 找不到文件，请确认 CSV 文件是否放在了 raw_data_sources 文件夹里。\n详细错误：{e}")