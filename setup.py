import os
from setuptools import setup, find_packages

setup(
    name='fire_chat',
    version='0.1.0',
    author='Collin M Brown',
    author_email='collin.brown.m@gmail.com',
    package_dir={"": "src"},  # Set the package directory to 'src'
    packages=find_packages(where="src"),  # Look for packages in 'src'
    url='http://pypi.python.org/pypi/MyPackageName/',
    license='LICENSE.txt',
    description='Modules for making chat with fireworks easy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['annotated-types==0.6.0', 'anyio==4.3.0', 'asttokens==2.4.1', 'build==1.2.1', 'certifi==2024.2.2', 'charset-normalizer==3.3.2', 'comm==0.2.2', 'debugpy==1.8.1', 'decorator==5.1.1', 'distro==1.9.0', 'exceptiongroup==1.2.1', 'executing==2.0.1', 'filelock==3.14.0', 'fire_chat @ file:///home/navi/Repos/fire_chat/dist/fire_chat-0.1.0.tar.gz', 'fireworks-ai==0.14.0', 'fsspec==2024.3.1', 'h11==0.14.0', 'httpcore==1.0.5', 'httpx==0.27.0', 'httpx-sse==0.4.0', 'huggingface-hub==0.23.0', 'idna==3.7', 'ipykernel==6.29.4', 'ipython==8.24.0', 'jedi==0.19.1', 'jupyter_client==8.6.1', 'jupyter_core==5.7.2', 'matplotlib-inline==0.1.7', 'nest-asyncio==1.6.0', 'numpy==1.26.4', 'openai==1.30.1', 'packaging==24.0', 'parso==0.8.4', 'pexpect==4.9.0', 'pillow==10.3.0', 'platformdirs==4.2.1', 'prompt-toolkit==3.0.43', 'psutil==5.9.8', 'ptyprocess==0.7.0', 'pure-eval==0.2.2', 'PyAudio==0.2.14', 'pydantic==2.7.1', 'pydantic_core==2.18.2', 'pygame==2.5.2', 'Pygments==2.18.0', 'pyproject_hooks==1.1.0', 'python-dateutil==2.9.0.post0', 'python-dotenv==1.0.1', 'PyYAML==6.0.1', 'pyzmq==26.0.3', 'regex==2024.4.28', 'requests==2.31.0', 'safetensors==0.4.3', 'six==1.16.0', 'sniffio==1.3.1', 'stack-data==0.6.3', 'tokenizers==0.19.1', 'tomli==2.0.1', 'tornado==6.4', 'tqdm==4.66.4', 'traitlets==5.14.3', 'typing==3.7.4.3', 'typing_extensions==4.11.0', 'urllib3==2.2.1', 'wcwidth==0.2.13', 'websockets==12.0'],
)