# 📄 PDF 编译指南

## 问题
服务器 (volta04) 无法访问外网，tectonic 无法下载 LaTeX 包。

## 解决方案

### 方案1: 本地编译 (推荐)

在你的 Mac/PC 上：

```bash
# 1. 克隆最新代码
git pull

# 2. 进入 theory 目录
cd theory

# 3. 编译 (任选一个)
pdflatex merged_theory.tex
# 或
tectonic merged_theory.tex
# 或
xelatex merged_theory.tex
```

### 方案2: Overleaf

1. 上传 `theory/merged_theory.tex` 到 Overleaf
2. 编译
3. 下载 PDF

### 方案3: 等待服务器网络恢复

```bash
cd theory
tectonic merged_theory.tex
```

## 已修正的文件

| 文件 | 需要重新编译 |
|------|-------------|
| `merged_theory.tex` | ✅ 是 |
| `theoretical_analysis.tex` | 被 merged 包含 |
| `appendix_proofs.tex` | 被 merged 包含 |

## 输出

编译后的新 PDF 将替换:
```
theory/IPS_Theoretical_Analysis.pdf
```
