from setuptools import find_packages, setup


package_name = "supermarket_scene_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "supermarket_scene_node = supermarket_scene_ros.supermarket_scene_node:main",
        ],
    },
)
