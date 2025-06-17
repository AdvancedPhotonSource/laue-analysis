from setuptools import setup, find_packages
from setuptools.command.build import build
import subprocess
import os
import shutil
from pathlib import Path

class CustomBuild(build):
    """Custom build command to compile C programs using their makefiles"""
    
    def run(self):
        # Compile our C programs first
        self.compile_c_programs()
        
        # Then run standard build process
        super().run()
    
    def compile_c_programs(self):
        """Compile C programs for each submodule separately"""
        # Check for required build tools
        required_tools = ['make', 'gcc']
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        # Special check for h5cc since it might not support --version
        try:
            result = subprocess.run(['h5cc', '-show'], capture_output=True, text=True)
            if result.returncode != 0:
                missing_tools.append('h5cc')
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append('h5cc')
        
        if missing_tools:
            print(f"Error: Missing required build tools: {', '.join(missing_tools)}")
            print("Please install the missing tools and try again.")
            return

        # Define submodules and their C programs
        submodules = {
            'indexing': {
                'programs': {
                    'euler': 'euler',
                    'peaksearch': 'peaksearch', 
                    'pixels2qs': 'pix2qs'  # This program creates 'pix2qs' executable
                },
                'special_builds': {'peaksearch': ['make', 'linux']}
            }
            # Future submodules can be added here
        }
        
        base_dir = Path(__file__).parent
        
        for submodule, config in submodules.items():
            print(f"\n=== Compiling {submodule} submodule ===")
            src_dir = base_dir / 'src' / 'laueanalysis' / submodule / 'src'
            bin_dir = base_dir / 'src' / 'laueanalysis' / submodule / 'bin'
            bin_dir.mkdir(exist_ok=True)
            
            print(f"Looking for source directories in: {src_dir}")
            
            for program_dir, exe_name in config['programs'].items():
                program_path = src_dir / program_dir
                if not program_path.exists():
                    print(f"Warning: Source directory for {program_dir} not found at {program_path}")
                    continue
                    
                print(f"Compiling {program_dir} in {program_path}...")
                
                # Check for special build commands
                if program_dir in config['special_builds']:
                    command = config['special_builds'][program_dir]
                else:
                    command = ['make']
                    
                result = subprocess.run(command, cwd=program_path, 
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
                        print(f"    Expected at: {exe_src}")
                        # List files in the directory to help debug
                        try:
                            files = list(program_path.iterdir())
                            print(f"    Files in {program_path}: {[f.name for f in files]}")
                        except Exception as e:
                            print(f"    Could not list directory: {e}")
                else:
                    print(f"  ✗ {program_dir} compilation failed (exit code {result.returncode})")
                    if result.stdout:
                        print(f"  Standard output:\n{result.stdout}")
                    if result.stderr:
                        print(f"  Error output:\n{result.stderr}")
                    # Don't fail the entire installation, just warn


setup(
    name='laueanalysis',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src', include=['laueanalysis', 'laueanalysis.*']),
    include_package_data=True,
    package_data={
        'laueanalysis.indexing': ['bin/*'],
    },
    cmdclass={
        'build': CustomBuild,
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