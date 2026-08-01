---
id: documentation-system-reorganization
title: RedRHex 文件系統重整計畫
lang: zh-TW
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-01
---

<a id="objective"></a>
## 目標

實作核准的雙語文件系統，同時不遺失可長期保存的知識、不破壞操作人員或開發人員的使用旅程，也不影響現有 panel 應用程式。各階段必須依序執行；來源文件的雙語替代文件與標題層級可追溯性通過驗證前，不得移除來源文件。

<a id="checkpoint-context"></a>
## 檢查點背景

來源為 commit `5de992d`。該 commit 記錄的基準測試結果如下：

- Reward Agent：`14` passed。
- Training panel：`208` passed。
- Sim-to-real：`528` passed。

這些結果須保留為檢查點證據；後續若有失敗，必須在完成前解釋或修正。

<a id="phase-1-isolation"></a>
## 階段 1 — 隔離與書面檢查點

- [ ] 從 `5de992d` 建立專用 worktree 與分支，並保留所有無關 worktree。
- [ ] 在來源 commit 建立 annotated tag `docs-reorg-source-2026-08-01`。
- [ ] 保存核准的英文與繁體中文設計配對，以及本實作計畫配對。
- [ ] 人工比較每組配對的非在地化 frontmatter、明確錨點 ID、限制、命令與驗收條件。
- [ ] 在開始遷移或實作工具前舉行書面規格審查檢查點並記錄核准結果。

**階段驗收：** 分支從正確來源隔離建立、無關 worktree 未變更、annotated source tag 指向 `5de992d`、四份正式檢查點文件均通過對等配對驗證，且書面核准已有紀錄。

<a id="phase-2-governance"></a>
## 階段 2 — 治理與目標檔案樹

- [ ] 建立雙語操作人員與開發人員入口及核准的中央目錄樹。
- [ ] 在治理文件定義 frontmatter schema、列舉值、生命週期與位置規則、命名規則、過時政策及文件影響政策。
- [ ] 為每種維護文件類型加入範本，並加入涵蓋意義對等與穩定明確錨點的翻譯指南。
- [ ] 定義根目錄與元件使用簡短雙語 `README.md` 的路由慣例。
- [ ] 建立標題層級遷移清單，欄位包括舊路徑、舊標題、處置方式、取代文件及移除 commit。
- [ ] 遷移或刪除來源檔前，將每份來源文件的每個標題加入清單。

**階段驗收：** 入口與治理文件均已配對並互相連結；schema 範例通過既定規則；路由與翻譯慣例明確無歧義；內容遷移前，每個範圍內的舊標題都有清單紀錄。

<a id="phase-3-validator"></a>
## 階段 3 — 以 TDD 建立驗證器、hooks 與 CI

- [ ] 每項驗證器行為都先撰寫失敗單元測試，再實作功能。
- [ ] 完全依照下列介面實作：

```text
python -m tools.documentation validate --all
python -m tools.documentation validate --staged
python -m tools.documentation validate --changed-from REF
python -m tools.documentation inventory --format json
python -m tools.documentation stage-site --output DIR
```

- [ ] 成功時回傳結束碼 `0`，驗證失敗時回傳 `1`。
- [ ] 驗證檔名、必要 frontmatter 與列舉值、ID 唯一性、生命週期與位置相容性、雙語配對是否存在及中繼資料是否一致、連結與錨點，以及變更配對一致性。
- [ ] 加入適合遷移、擁有權、過時情況及 CI 證據使用的清冊報告。
- [ ] Pre-commit 執行 `validate --staged`，CI 執行 `validate --all`。
- [ ] 要求結構化 PR 欄位 `Docs impact: none | operator | developer | shared | release | experiment` 與 `Docs reason: <required explanation>`。
- [ ] 強制宣告存在且格式正確，但不從原始碼路徑推論語意影響。

**階段驗收：** 測試證明錯誤檔名、缺少或無效 frontmatter、重複 ID、生命週期與位置不相容、缺少配對、配對中繼資料偏移、只改一種語言、損壞連結、缺少錨點及錨點不一致時均會失敗；所有正向 fixture 通過；CLI 結束碼符合契約；清冊輸出可由機器讀取；pre-commit 與 CI 關卡快速且均衡，不重複執行高成本且無關的測試套件。

<a id="phase-4-central-migration"></a>
## 階段 4 — 中央文件遷移

- [ ] 將 `docs/COMMANDS.md` 與 `docs/redrhex_train_play_guide.md` 拆分為雙語的操作人員設定、訓練、評估、匯出影片及疑難排解指南，以及共用命令與版本參考。
- [ ] 每個遷移後的命令、版本及路徑都必須對照程式碼驗證後才發布。
- [ ] 從期中、會議、訓練、ForwardFast、改善、能量及七月審查報告中，擷取可長期保留的架構、決策、已驗證結果、引用資料與未解工作。
- [ ] 能量相關資料保留已驗證的能量模型、獎勵設計理由、限制與驗證；捨棄快照及重複的 LaTeX 附錄。
- [ ] 將 sim-to-real 內容拆分為操作人員校正指南與開發人員證據架構。
- [ ] 更新每個清單紀錄的處置方式與雙語替代文件連結。
- [ ] 只有當雙語替代文件、連結、錨點、命令及可追溯性全部通過驗證後，才移除原始檔，並在清單記錄移除 commit。

**階段驗收：** 完整雙語操作人員旅程涵蓋設定、訓練、評估、影片匯出、校正、部署到疑難排解；完整雙語開發人員旅程涵蓋架構、開發、測試、子系統、決策與證據；命令、版本及路徑與程式碼相符；每個被移除標題都有遷移清單紀錄；任何重複原始內容都不得在沒有明確處置方式的情況下保留。

<a id="phase-5-project-components"></a>
## 階段 5 — 專案與元件遷移

- [ ] 將已實作的 Reward Agent 基礎與權重調校知識移至維護文件；將未完成的 proposal UI 與 panel 整合工作列入路線圖。
- [ ] 將 Windows 遠端啟動器設計與計畫遷移為雙語現行配對。
- [ ] 保留 reboot 分支，並將其獨有內容標為 `proposed`，而非 `active`。
- [ ] 產出雙語配對的 panel 文件、操作手冊及彙整證據的 `3.4.10` 版本；不得虛構 `3.4.4` 到 `3.4.9`。
- [ ] 將單體 ROS README 拆為並置的雙語架構、政策契約、bring-up、部署與疑難排解文件，並保留簡短雙語路由 `README.md`。
- [ ] 確認未使用的 package changelog stub 沒有長期內容後將其移除。
- [ ] 整個遷移期間都不得把已忽略的執行期與產生式成品加入 Git。

**階段驗收：** 元件入口連結每組並置文件；已實作與未完成工作正確分離；reboot 提案仍可復原；panel 版本有證據且版號正確；ROS 路由可抵達每份雙語指南；未使用 stub 已記入清單；Git 不含產生式或執行期成品。

<a id="phase-6-skills"></a>
## 階段 6 — 依序測試儲存庫技能

- [ ] 一次實作一個 `writing-redrhex-docs` 與 `reviewing-redrhex-docs`；兩者都引用治理文件而不複製政策。
- [ ] 每個技能先記錄未使用技能的 RED 基準，再記錄使用技能的 GREEN 執行結果。
- [ ] 檢查失敗與漏洞、調整技能，並重跑相同情境。
- [ ] 每個技能至少重複五次微型測試，並設置無指引對照組。
- [ ] 情境套件整體涵蓋 operator、developer、release、ADR、design、plan、experiment 及 no-impact 案例。

**階段驗收：** 兩個技能相對無指引對照組均呈現可重現的改善；各自至少有五次微型測試紀錄；每個必要情境類型皆經測試；漏洞修正有文件紀錄；輸出遵守治理規則且未複製政策。

<a id="phase-7-site"></a>
## 階段 7 — 雙語網站

- [ ] 鎖定 MkDocs、Material for MkDocs 與 `mkdocs-static-i18n` 相依套件版本。
- [ ] 為 `.en.md` 與 `.zh-TW.md` 設定後綴式在地化。
- [ ] 納入版本控制的暫存清單同時包含中央與並置文件。
- [ ] 使用 `stage-site --output DIR` 可重現地暫存文件；產生的 HTML 維持忽略狀態。
- [ ] 保留 panel UI 於遠端 Pages 根目錄，文件發布在 `/docs/` 下。
- [ ] 測試嚴格建置、內部與跨語言連結、對等頁面語言切換及雙語搜尋。

**階段驗收：** 兩種語言網站都能嚴格建置；每個對等頁面都能正確切換；英文與繁體中文內容都可搜尋；中央與並置頁面都存在；損壞連結會使建置失敗；Git 不含產生的 HTML；Pages 根目錄的現有 panel 應用程式保持不變。

<a id="phase-8-final-checkpoint"></a>
## 階段 8 — 最終檢查點

- [ ] 執行完整的清冊、連結、網站、遷移及過時報告，並保留輸出作為審查證據。
- [ ] 執行受文件工具影響的儲存庫測試，並將相關結果與來源檢查點比較。
- [ ] 確認所有遷移清單紀錄都有最終處置方式，所有被移除標題都有移除 commit。
- [ ] 確認工作樹不含產生的網站輸出、執行期成品或未記錄的來源文件刪除。
- [ ] 取得最終審查後，在核准 commit 建立 annotated tag `docs-reorg-v1`。
- [ ] Git 歷史與分支清理視為日後另行審查的專案；不得納入本計畫執行。

**階段驗收：** 所有報告都乾淨或其發現已明確接受；測試通過；雙語旅程完整；清單可完全追溯；panel 應用程式保持不變；最終 annotated tag 指向經審查結果；清理工作未混入此變更。

<a id="overall-acceptance"></a>
## 整體驗收條件

- [ ] 驗證器失敗 fixture 涵蓋命名、中繼資料與列舉值、ID 唯一性、生命週期與位置、配對與一致性、變更配對一致性、連結及錨點，並回傳 `1`；有效輸入回傳 `0`。
- [ ] 每個從舊文件移除的標題都有遷移清單紀錄，包含處置方式、雙語替代文件及移除 commit。
- [ ] 每個發布的命令、版本及路徑均已對照程式碼驗證。
- [ ] 操作人員的設定、訓練、評估、影片匯出、校正、部署及疑難排解在兩種語言都形成完整旅程。
- [ ] 開發人員的架構、開發、測試、子系統、決策及證據文件在兩種語言都形成完整旅程。
- [ ] 技能測試涵蓋 operator、developer、release、ADR、design、plan、experiment 及 no-impact 情境，包括 RED/GREEN 證據、漏洞調整、至少五次重複及無指引對照組。
- [ ] 兩種網站語言都能成功建置、連結、切換對等頁面並搜尋，同時 panel 應用程式在根目錄保持不變。
- [ ] Pre-commit 與 CI 關卡強制文件契約，但不過度重複工作，也不推論語意影響。
- [ ] Git 不含產生的 HTML、暫存網站輸出、快取、日誌、影片或其他執行期成品。
