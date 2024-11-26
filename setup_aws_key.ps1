# Rename the key file
Rename-Item -Path "aws_key.pem.pem" -NewName "aws_key.pem" -Force

# Reset permissions
icacls.exe "aws_key.pem" /reset

# Remove inheritance
icacls.exe "aws_key.pem" /inheritance:r

# Set read-only for current user
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
icacls.exe "aws_key.pem" /grant:r "${currentUser}:(R)"

Write-Host "AWS key file has been set up with correct permissions."
