---
id: readme-router-convention
title: README 路由慣例
lang: zh-TW
audience: developer
type: reference
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="exception"></a>
## 雙語路由例外

根目錄與元件 `README.md` 是簡短的單檔雙語路由，也是面向人員文件不使用 `.en.md` 與 `.zh-TW.md` 配對的唯一例外。路由長度應以約一個畫面為上限，協助新讀者選擇下一份正式文件。

<a id="root-router"></a>
## 根目錄路由

根目錄 `README.md` 必須：

- 以英文與繁體中文說明 RedRHex；
- 分別提供英文與繁體中文的操作人員及開發人員入口連結；
- 避免容易變動的命令、版本、設定值及持續維護的程序細節。

<a id="component-router"></a>
## 元件路由

元件 `README.md` 必須以雙語說明元件用途與擁有者，並在相關類型存在時連結至並置的操作人員、開發人員、參考及版本文件。

<a id="knowledge-boundary"></a>
## 知識邊界

持續維護的詳細知識應置於正式雙語配對檔，不得放入 README。路由只能包含足以辨識目的地的穩定背景，不得成為另一份政策、設定指南、架構說明或變更紀錄。

<a id="review-checklist"></a>
## 審查清單

- 兩種語言說明相同專案或元件及擁有者。
- 操作人員與開發人員目的地容易區分。
- 語言連結抵達對等的正式頁面，且相對連結有效。
- 不含易變動的命令、版本、設定或持續維護的細節知識。
- 路由維持約一個畫面。
