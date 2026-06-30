from setuptools import find_packages, setup

package_name = 'global_planer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chirag',
    maintainer_email='chirag@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "map_gen_ = global_planer.map_pre_prcoessor:main",
            "spawn_sim_vehicle = global_planer.sim_vehicle_tf_generator:main",
            "hybrid_a_star_ = global_planer.hyprid_A_star:main",
            "spwan_real_vehicle = global_planer.real_robot_command_and_odom_relay:main",
            "data_collect = global_planer.data_recorder:main"
        ],
    },
)
