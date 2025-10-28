# build.py - VERSIÓN MEJORADA PARA PORTABILIDAD

import PyInstaller.__main__
import os
import sys
from pathlib import Path

def build_executable():
    project_root = Path(__file__).parent
    
    main_script = project_root / "interfaz" / "ventana_principal.py"
    
    if not main_script.exists():
        print(f" Error: No se encuentra {main_script}")
        return

    # Argumentos COMUNES para todas las plataformas
    args_comunes = [
        str(main_script),
        "--onefile",
        "--name=ANFIS_Tumor_Cerebral",
        "--add-data=config:config",
        "--add-data=core:core", 
        "--add-data=utils:utils",
        "--add-data=interfaz:interfaz",
        "--hidden-import=sklearn",
        "--hidden-import=scipy",
        "--hidden-import=PIL",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.backends.backend_tkagg",
        "--hidden-import=tkinter",
        "--exclude-module=tkinter.test",
        "--exclude-module=matplotlib.tests",
        "--clean"
    ]
    
    # Argumentos ESPECÍFICOS por plataforma
    if sys.platform == "win32":
        args_comunes.append("--windowed")
        args_comunes.append("--icon=assets/icono.ico")  # Si tienes icono
    elif sys.platform.startswith("linux"):
        args_comunes.append("--windowed")
        # Linux puede necesitar hooks adicionales
        args_comunes.extend([
            "--hidden-import=xlib",
            "--hidden-import=gi",
        ])
    
    # Rutas de build
    args_comunes.extend([
        f"--workpath={project_root / 'build'}",
        f"--distpath={project_root / 'dist'}", 
        f"--specpath={project_root}"
    ])
    
    print(f" Construyendo para: {sys.platform}")
    print(f" Script principal: {main_script}")
    
    PyInstaller.__main__.run(args_comunes)

if __name__ == "__main__":
    build_executable()