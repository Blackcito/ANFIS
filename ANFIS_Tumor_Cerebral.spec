# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Modelo_anfis_ajustado/interfaz/ventana_principal.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config'), ('core', 'core'), ('utils', 'utils'), ('Modelo_anfis_ajustado', 'Modelo_anfis_ajustado')],
    hiddenimports=['sklearn', 'scipy', 'PIL', 'matplotlib', 'tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ANFIS_Tumor_Cerebral',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
