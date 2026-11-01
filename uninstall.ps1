# Удаление «Диктовки»: выключает программу, убирает ярлыки и папку
$Dir = $PSScriptRoot
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like "*diktovka.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
foreach ($p in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Startup'))) {
  Remove-Item (Join-Path $p 'Диктовка.lnk') -ErrorAction SilentlyContinue
}
Set-Location $env:TEMP
Remove-Item $Dir -Recurse -Force
Write-Host 'Диктовка удалена.' -ForegroundColor Green
