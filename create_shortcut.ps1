$SourceFilePath = "C:\Users\ZhuanZ\visual_search_project\visual search copy3\run_forever.bat"
$ShortcutPath = [System.Environment]::GetFolderPath("Startup") + "\VisualSearchServer.lnk"

$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $SourceFilePath
$Shortcut.WorkingDirectory = "C:\Users\ZhuanZ\visual_search_project\visual search copy3"
$Shortcut.WindowStyle = 7  # Minimized window
$Shortcut.Description = "Start Visual Search Server"
$Shortcut.Save()

Write-Host "Shortcut created successfully at: $ShortcutPath"
