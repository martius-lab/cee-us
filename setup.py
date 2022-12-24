from setuptools import setup

package_name = "mbrl"

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [
            "main = " + package_name + ".main:main",
        ],
    },
)
