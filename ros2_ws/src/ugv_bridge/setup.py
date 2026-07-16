import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ugv_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="path_plannig_pfc",
    maintainer_email="you@example.com",
    description=(
        "Robot-side ROS 2 package: receives the annotated path image and "
        "nav_msgs/Path from path_plannig_pfc and drives a simulated UGV."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "path_follower_node = ugv_bridge.path_follower_node:main",
            "image_monitor_node = ugv_bridge.image_monitor_node:main",
            "tf_odom_bridge_node = ugv_bridge.tf_odom_bridge_node:main",
        ],
    },
)
