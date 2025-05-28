from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import shutil
from pathlib import Path

class CustomBuildExt(build_ext):
    """Custom build command to compile C programs using their makefiles"""
    
    def run(self):
        # First run the standard extension building
        super().run()
        
        # Then compile our C programs
        self.compile_c_programs()
    
    def compile_c_programs(self):
        """Compile the C programs using their existing makefiles"""
        # Map program directories to their executable names
        programs = {
            'euler': 'euler',
            'peaksearch': 'peaksearch', 
            'pixels2qs': 'pix2qs'  # This program creates 'pix2qs' executable
        }
        
        base_dir = Path(__file__).parent
        index_src = base_dir / 'index_src'
        
        # Create bin directory in the package
        bin_dir = base_dir / 'laueindexing' / 'bin'
        bin_dir.mkdir(exist_ok=True)
        
        for program_dir, exe_name in programs.items():
            program_path = index_src / program_dir
            if not program_path.exists():
                print(f"Warning: Source directory for {program_dir} not found")
                continue
                
            print(f"Compiling {program_dir}...")
            try:
                # Run make in the program directory
                result = subprocess.run(['make'], cwd=program_path, 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Copy the executable to the package bin directory
                    exe_src = program_path / exe_name
                    exe_dst = bin_dir / exe_name
                    
                    if exe_src.exists():
                        shutil.copy2(exe_src, exe_dst)
                        exe_dst.chmod(0o755)
                        print(f"  ✓ {program_dir} compiled successfully")
                    else:
                        print(f"  ✗ {exe_name} executable not found after compilation")
                else:
                    print(f"  ✗ {program_dir} compilation failed (exit code {result.returncode})")
                    if result.stderr:
                        print(f"  Error output:\n{result.stderr}")
                        
            except Exception as e:
                print(f"  ✗ Unexpected error compiling {program_dir}: {e}")

setup(
    name='laueindexing',
    version='0.1.0',
    packages=find_packages(include=['laueindexing', 'laueindexing.*']),
    include_package_data=True,
    package_data={
        'laueindexing': ['bin/*'],
    },
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    description='A package for Laue indexing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/laue_indexing',
    install_requires=[
        'numpy',
        'pyyaml',
    ],
    python_requires='>=3.12',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
    ],
)