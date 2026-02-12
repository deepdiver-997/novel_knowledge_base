from setuptools import setup, find_packages

setup(
    name="novel-knowledge-base",
    version="0.1.0",
    description="Novel knowledge base with MCP stdio server",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "requests",
        "mcp",
    ],
    entry_points={
        "console_scripts": [
            "novel-kb=novel_kb.main:main",
        ]
    },
)
