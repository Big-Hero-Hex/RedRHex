---
id: project-management-sync
title: 專案管理同步
lang: zh-TW
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-15
---

<a id="outcome"></a>
## 成果

此 contract 讓 RedRHex 專案管理 agent 能看見 repository 工作，同時避免為 maintained documentation 建立第二份可編輯副本。Agent 直接讀取 [GitHub](https://github.com/Big-Hero-Hex/RedRHex)，將協調狀態投影到 [PM Control Center](https://docs.google.com/spreadsheets/d/1DC4RlA1jVbcLEFDr0Jom1bysjc8cBxX8w_8d7uv0zjs/edit)，並連結 [shared Drive](https://drive.google.com/drive/folders/1REUOdoJO_CHnxuBSGBHZwVIgA1KFW7gJ) 中的證據。

<a id="prerequisites-and-context"></a>
## 前置條件與背景

- PM agent 需要 GitHub repository 的讀取權限，以及 PM Control Center 的寫入權限。
- Scheduled prompt 或 webhook receiver 必須能喚起 PM agent。Repository 或 Drive credential 不得 commit 到 GitHub，也不得寫入專案文件。
- Repository default branch 是 `main`。討論 code 與 documentation revision 時，以 immutable commit SHA 識別精確版本。
- PM agent 的核准邊界維持不變：可以協調例行紀錄並提出工作建議，但沒有明確 human approval 時，不得授權 source change、deployment、budget、publication claim、autonomous campaign arming 或 hardware activity。

<a id="instructions-or-explanation"></a>
## 指示或說明

每一類專案事實只使用一個 authoritative home：

| 專案事實 | Authoritative home | 其他系統的處理方式 |
| --- | --- | --- |
| Code、configuration、test、maintained documentation、plan、design、decision、roadmap 與 release | Immutable SHA 所指向的 GitHub `main` | Control Center 保存狀態與 permalink，不複製 canonical text。 |
| 已接受的 task、human DRI、date、status、decision、risk、meeting 與 paper readiness | PM Control Center | 必要時由 repository 與 Drive 文件連到相關紀錄。 |
| Meeting record、hardware measurement、CAD/manufacturing material、experiment package、video、report 與 paper asset | Shared Drive | Control Center 保存 evidence link 與 readiness state。 |
| Run identity、metric、checkpoint、log 與 evaluation evidence | Training Panel 或 experiment database | Control Center 保存 immutable evidence link 與 code SHA。 |

Drive 中的 maintained repository Markdown 副本只可作為方便閱讀的 snapshot。Snapshot 必須顯示 GitHub source URL、branch 或 tag、commit SHA 與 refresh time。當 SHA 與 `main` 不同時，不得用它決定目前 scope 或關閉 task。對於會反覆更新的內容，應以 GitHub link 取代 snapshot link，而不是持續手動複製。

每次 repository intake 依下列流程執行：

1. 讀取目前的 `refs/heads/main` SHA。尚未 merge 的 branch、recovery snapshot 與 pull request 只能視為 candidate work，不得回報為已發布或目前狀態。
2. 將 SHA 與上次成功 ingest 的 SHA 比較。SHA 沒有變更時不執行任何寫入。
3. 讀取中間的 commit 與 changed path。`docs/roadmap/`、`docs/plans/active/`、`docs/designs/active/`、`docs/decisions/` 與 `docs/releases/` 下的變更一律需要 PM review。Source、test、workflow 與 configuration 變更若影響 deliverable、gate、risk、evidence、compatibility 或 date，也需要 review。
4. 為新的 SHA 或連續 commit range 在 `Update Log` append 一筆 deduplicated 紀錄。記錄 SHA、changed canonical path、observed fact、interpretation 或 task proposal、blocker、evidence permalink、confidence 與 `Pending` review state。不可只因 code 存在就推斷工作已完成。
5. 執行 PM reconciliation。只有 evidence 支持時才更新 `Tasks`、`Experiments`、`Decisions`、`Risks`、`Roadmap` 或 `Meetings`。每個 actionable task 必須有一位 human DRI、明確日期、acceptance criteria、dependency 與 evidence destination。
6. 只有在 Control Center 寫入成功，且 read-back 確認更新後，才保存新的 intake cursor。Retry 依 repository identity、branch 與 commit SHA deduplicate。

建議先以 scheduled polling 作為 notification path，因為不需要 inbound endpoint：每次 PM sync 開始時，以及專案選定的 active-project cadence 檢查 GitHub。只有在 PM agent 具備 authenticated endpoint、retry handling、deduplication 與 secret rotation 後才改用 GitHub webhook。Webhook 只負責喚起 agent；agent 仍以 GitHub 為讀取來源。

Documentation intake 遵循 repository 的[文件治理](../governance/index.zh-TW.md)。Meaning-changing edit 必須同時更新兩個 locale file；PM agent 應使用 canonical path 與 anchor link，不把內文複製到 Drive。

<a id="verification"></a>
## 驗證

只有下列檢查全數通過時，integration 才算正常運作：

- Control Center 記錄 repository URL、default branch，以及最近觀察到與最近成功 ingest 的 `main` SHA。
- 一個新的 `main` SHA 只產生一筆含 GitHub permalink 的 pending Update Log 紀錄；再次 polling 不會產生重複紀錄。
- Control Center 寫入失敗時 intake cursor 不變；retry 後會產生缺少的紀錄。
- 只存在 branch 的 documentation 或 source change 只描述為 proposed 或 in progress，絕不描述為 shipped。
- Merge 到 `main` 的 plan change 會成為可 review 的 PM update；被接受的工作會轉成包含 owner、date、acceptance criteria、dependency 與 evidence 的個人 task packet。
- 過期 Drive snapshot 會由其 SHA 或缺少 provenance 被辨識出來，而且不會被視為目前狀態。

截至 2026-08-15，Control Center task `T-003` 與 `T-020` 分別追蹤 repository identity 與 intake automation。Update `UPD-20260815-REPO-001` 是第一筆 pending repository audit record。

<a id="troubleshooting-and-limits"></a>
## 疑難排解與限制

- 若無法讀取 GitHub，將 repository sync 回報為 Yellow 或 Red，且不得推進 cursor。
- 若無法寫入 Drive 或 Control Center，保留 pending change packet 並 retry；不得聲稱 reconciliation 已成功。
- 若兩個系統的內容衝突，依上方 authoritative-home table 判定，並將 mismatch 記錄成 correction 或 blocker。
- Repository polling 只能偵測已 commit 到 GitHub 的狀態，無法發現尚未 push 的 local working tree；local work 在 push 或 merge 前必須明確標示為 uncommitted。
- 此 contract 定義 intake 與 reconciliation，但不會自行 provision GitHub connector、scheduled prompt、webhook endpoint 或 credential；`T-020` 負責關閉這項 implementation gap。

