# RVC-Vulkan 主题包机制（预留扩展）

> 网页前端支持**主题包**扩展：任意第三方或用户可在 `runtime/static/themes/` 下
> 添加自己的主题目录，前端"主题"下拉会自动发现并切换，无需改代码。

## 主题包结构

```
runtime/static/themes/
├── default.css            # 默认主题（CSS 变量，基座）
├── README.md              # 本说明
└── <主题名>/              # 你的主题目录（目录名即主题 id）
    ├── theme.css          # 覆盖 CSS 变量（必填）
    └── theme.json         # 元数据：{"name":"显示名","author":"作者","desc":"说明"}(可选)
```

## 怎么写一个主题

1. 新建目录 `runtime/static/themes/mytheme/`；
2. 写 `theme.css`：只需覆盖你要改的 CSS 变量，其余继承默认主题——

```css
:root {
    --bg: #f4f6fb;        /* 浅色背景 */
    --card: #ffffff;
    --fg: #1f2937;
    --muted: #6b7280;
    --accent: #10b981;    /* 绿松石强调色 */
    --border: #e5e7eb;
}
```

3. （可选）写 `theme.json`：`{"name":"我的主题","author":"Alice","desc":"浅色护眼主题"}`；
4. 重启网页后端，右上角"主题"下拉即可看到并切换。

## 预置扩展点（为后续功能预留）

- **中英文文案**：页面文案集中在 `index.html` 顶部的 `I18N` 字典，后续可扩展为
  `themes/<主题>/i18n_zh.json` 等按语言加载（当前默认中文）；
- **自定义样式类**：默认主题只改 CSS 变量；主题内可追加 `.card {}`、`#result audio {}`
  等覆盖细节样式；
- **Logo / 背景图**：可在主题目录放 `logo.png` / `bg.jpg`，通过主题 css 引用
  `url(./logo.png)`（相对主题目录）。

## 说明

- 主题仅影响网页外观，不影响推理/训练功能；
- 切换主题不需要重新加载模型；
- 默认主题 `default.css` 是全部 CSS 变量的唯一权威定义，其它主题都是覆盖它的增量。
