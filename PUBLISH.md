# 发布步骤(从本机推到 GitHub)

本地已就绪:LICENSE / README(结果速览 + roadmap)/ .gitignore(已排掉 models 5.5G、
data_raw 34M、实验二进制,报告与 JSON 会入库)。以下命令按顺序执行即可。

## 1. 在 GitHub 建空仓库

浏览器打开 https://github.com/new
- Repository name: `adaptive-retrieval`
- **不要**勾选 "Add a README / .gitignore / license"(本地已备好)
- Public

## 2. 本地初始化并提交(在仓库根目录执行)

```bash
cd "D:\claude code use\better research skill"

git init -b main
git add -A
git status --short          # 检查:不应出现 models/ data_raw/ *.npy *.pkl
git commit -m "adaptive-retrieval: 一致性路由 + 语料感知档位, M1-M7 评测闭环, 跨 harness Skill 打包"
```

## 3. 关联远程并推送

```bash
git remote add origin https://github.com/linchengran/adaptive-retrieval.git
git push -u origin main
```

推送时 Windows 凭据管理器会弹出 GitHub 登录(你之前推过 sentiment-board,
凭据大概率已在;若弹窗就登录一次)。

## 4. 推完之后

- 仓库页 → About 填一句话描述,Topics 加 `retrieval` `rag` `agent-skills` `bm25` `reranker`
- 把主页(D:\git项目\new try\index.html)PROFILE.works 第一条的 links.href
  改成 https://github.com/linchengran/adaptive-retrieval —— 卡片自动点亮
- 后续把 评审文档 里的 P0/P1/P2 逐条建成 GitHub Issues,README 的路线图就活了

## 备注

- 若推送报错 large files:说明有漏网大文件,`git rm --cached <文件>` 后重commit,
  并把模式补进 .gitignore
- 若走 SSH:`git remote add origin git@github.com:linchengran/adaptive-retrieval.git`
