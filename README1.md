# adolescent-bmi-prediction

代码与示例数据，用于复现实验中关于青少年 BMI 预测的分析与图表。

## Quick Start

```bash
# 1) 安装依赖（Python 3.11.3）
pip install -r requirements.txt
# 如暂时没有 requirements.txt，可先按下列核心依赖安装
# pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 shap==0.46.0
# （若使用 CatBoost/LightGBM，请额外安装：pip install catboost==1.2.7 lightgbm==4.5.0）

# 2) 运行脚本（如需可在脚本内调整路径）
python src/BMIPre_1.py
python src/BMIPre_2.py
python src/BMIPre_3.py
```

## Dependencies

- Python 3.11.3  
- numpy==1.26.4  
- pandas==2.2.3  
- scikit-learn==1.5.2  
- shap==0.46.0  
- （可选）CatBoost 1.2.7，LightGBM 4.5.0（若脚本中启用）

> 建议在仓库根目录提供 `requirements.txt`，内容至少包含上述版本。

## Data

仓库已包含用于复现的脱敏/示例数据：
- `data/BMIPre_1_2.csv`
- `data/BMIPre_3.csv`

> 字段说明与预处理细节见脚本注释；若需原始数据的获取方式，请参阅论文的 Data Availability Statement 或联系作者。

## Methods (brief)

- **划分与验证**：80/20 训练–测试划分，**5×5 嵌套交叉验证（nested CV）** 进行调参与性能估计。  
- **异方差性与校准**：在**训练集 out-of-fold 残差**上进行 **Breusch–Pagan** 检验（α = 0.05）。若显著，则在训练端学习 **WLS 校准** 并**一次性应用**于测试集（不在测试集重拟合）。  
- **分层性能分析**：按关键亚组（如性别、年龄组、基线 BMI 分类）报告性能指标。  
- **误差分析**：包含带平滑线的预测值 vs. 实际值散点图、Bland–Altman 分析与误差分布。  
- **模型解释**：采用 **SHAP**（summary、dependence、interactions、waterfall）。

## Generated Figures

- 预测值 vs. 实际值散点图（含平滑线）  
- **Bland–Altman** 图  
- **SHAP** 图（summary、dependence、interactions、waterfall）

## Repository Layout

```
adolescent-bmi-prediction/
├─ src/
│  ├─ BMIPre_1.py
│  ├─ BMIPre_2.py
│  └─ BMIPre_3.py
├─ data/
│  ├─ BMIPre_1_2.csv
│  └─ BMIPre_3.csv
├─ README.md
├─ LICENSE
└─ requirements.txt  # 建议提供
```

## License

- **Code:** GNU General Public License v3.0 (GPL-3.0)  
- **Data:** 用于学术研究/非商业用途（如适用可在此注明 CC BY-NC 4.0）

## Citation

如使用本仓库，请引用对应论文与本仓库（建议创建版本发布，如 v1.0.0）。

## Contact

Issues on GitHub / Email: <your-email@example.com>
