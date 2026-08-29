@echo off
rem ============================================================
rem vcp-memo 记忆库备份(Windows 侧;WSL 通过 \\wsl$ 映射访问)
rem ============================================================
rem 用法:双击运行,或注册 schtasks 计划任务(见 README「运维·备份与恢复」节)。
rem 只镜像 diaries/(真相源:人工与插件共同写入,必须备份);
rem index/ 是派生产物,不备份 —— 恢复后删掉 index/ 目录,下次启动自动全量重建。
rem 数据目录默认 /home/lyy/vcp-memo-data(与 cordis.patch.yml 一致);
rem 若 dataRoot 或 WSL 发行版名改了,请同步修改下面的 SRC。
setlocal

rem 源:WSL 发行版 Arch 内的 diaries 目录(经 \\wsl$ 网络映射访问)。
set SRC=\\wsl$\Arch\home\lyy\vcp-memo-data\diaries
rem 目标:插件目录上两级下的 backup\vcp-memo-data\diaries
rem (即 <vcp-memo/..>/backup/...;脚本在 vcp-memo/scripts/ 下)。
set DST=%~dp0..\..\backup\vcp-memo-data\diaries
rem 可用环境变量覆盖目标目录:set VCP_MEMO_BACKUP_DST=D:\backup\vcp-memo\diaries
if defined VCP_MEMO_BACKUP_DST set DST=%VCP_MEMO_BACKUP_DST%

rem ── 友好报错:源目录不可达(WSL 未启动 / 发行版名不符 / dataRoot 变更)──
if not exist "%SRC%" (
  echo [错误] 备份源不存在或不可达: %SRC%
  echo        请确认:WSL 发行版名为 Arch 且已启动;dataRoot 为 /home/lyy/vcp-memo-data。
  echo        可在资源管理器地址栏输入 %SRC% 验证映射是否可访问。
  exit /b 2
)

rem ── 目标目录不存在时先建好,避免 robocopy 直接报错 ──
if not exist "%DST%" mkdir "%DST%"

rem robocopy 镜像备份:
rem   /MIR  镜像模式(目标以源为准,删除目标多余的旧文件)
rem   /R:2 /W:5 复制失败重试 2 次、每次间隔 5 秒
rem   /NFL /NDL /NP 不打印具体文件/目录名、不显示百分比进度(日志更干净)
rem   /LOG+ 追加写入脚本同目录 backup.log
robocopy "%SRC%" "%DST%" /MIR /R:2 /W:5 /NFL /NDL /NP /LOG+:"%~dp0backup.log"
set RC=%ERRORLEVEL%

rem robocopy 退出码说明:
rem   0-7  均为成功(0=无变化;1=有文件被复制;2=目标有额外文件;3 及以上为组合;
rem        且 8 以下都有成功备份发生)。
rem   8+   复制失败/源或目标不可达/参数错误,需要人工检查 backup.log。
echo 备份完成: %DATE% %TIME% (robocopy 退出码 %RC%) >> "%~dp0backup.log"

if %RC% GEQ 8 (
  echo [错误] robocopy 返回退出码 %RC% (大于等于 8 表示复制失败)。
  echo        请查看备份日志: %~dp0backup.log
  exit /b %RC%
)

echo 备份完成: diaries 已镜像到 %DST%
exit /b 0