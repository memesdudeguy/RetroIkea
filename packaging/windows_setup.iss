#define MyAppName "retro ikea"
#define MyAppVersion "Beta 3"
#define MyAppPublisher "NightShift"
#define MyAppExeName "RetroIkea.exe"

; Repo root containing `assets/` and `build-win-mingw/` (same tree as CMake VULKAN_GAME_ASSETS_SOURCE_DIR).
; Default: parent of this `packaging/` folder. Override: ISCC /DRetroIkeaRepoRoot="C:\full\path\RetroIkea"
; Wine example: Z:\home\<user>\Downloads\RetroIkea  (maps to /home/<user>/Downloads/RetroIkea)
#ifndef RetroIkeaRepoRoot
#define RetroIkeaRepoRoot ".."
#endif

[Setup]
AppId={{A7E45D0A-88C8-4AE4-A8F0-5D9A5D8E7B13}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=RetroIkea-Beta-Setup
Compression=lzma
SolidCompression=yes
; IKEA-inspired palette: blue #0058AB + yellow #FFCC00 (WizardBackColor etc. need Inno Setup 6.3+).
WizardStyle=modern windows11 light
WizardBackColor=#FAFCFE
WizardBackColorDynamicDark=#152838
WizardImageBackColor=#FFCC00
WizardImageBackColorDynamicDark=#C9A000
WizardSmallImageBackColor=#0058AB
WizardSmallImageBackColorDynamicDark=#003D73
ArchitecturesInstallIn64BitMode=x64compatible
; Icons/bitmaps live next to this .iss (see packaging/). Regenerate BMPs: python packaging/generate_setup_wizard_bmps.py
SetupIconFile=setup_icon.ico
WizardImageFile=setup_wizard.bmp
WizardSmallImageFile=setup_wizard_small.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "installtailscale"; Description: "Install Tailscale (recommended for WAN co-op: share your Tailscale tailnet)"; GroupDescription: "Networking (optional):"; Flags: unchecked

[Files]
; Adjust RetroIkeaRepoRoot (or {#RetroIkeaRepoRoot}\build-win-mingw) if your Windows build dir differs.
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; MinGW SDL2_image (Arch) links WebP: bundle DLLs if present next to the exe after build.
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libwebp.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libwebpdemux.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libsharpyuv.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libssp-0.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; SDL2_image loads PNG/JPEG via these at runtime (same idea as WebP DLLs).
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libpng16-16.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libjpeg-8.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; When built with -DVULKAN_GAME_WINDOWS_ALL_DYNAMIC=ON (shared Assimp + dynamic MinGW C++ runtime).
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libassimp-5.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libgcc_s_seh-1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libstdc++-6.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\libwinpthread-1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; Pipeline cache is read next to the exe (see VULKAN_GAME_PIPELINE_CACHE).
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\vulkan_game_pipeline.cache"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; Full tree from repo assets/ (e.g. .../RetroIkea/assets), then overlay Windows-built SPIR-V only.
; Game loads SHADER_DIR as assets/shaders/*.spv next to the exe; do not drop *.spv in {app} root.
Source: "{#RetroIkeaRepoRoot}\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#RetroIkeaRepoRoot}\build-win-mingw\assets\shaders\*"; DestDir: "{app}\assets\shaders"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#RetroIkeaRepoRoot}\packaging\third_party\tailscale-setup-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\tailscale-setup-amd64.exe"; Parameters: "/S"; StatusMsg: "Installing Tailscale…"; Tasks: installtailscale; Flags: skipifdoesntexist waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
