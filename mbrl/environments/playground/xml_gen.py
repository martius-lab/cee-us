base = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="playground">
    <compiler coordinate="local" angle="radian" inertiafromgeom="true"/>
    <!-- <option timestep="0.002" gravity="0 0 -9.81" impratio="1" integrator="Euler" iterations="20" tolerance="1e-8"/>-->
    <option timestep="0.002" integrator="Euler"/>
    <default>
        <joint armature="0" damping="0" limited="true"/>
        <geom contype="1" conaffinity="1" rgba="1 1 1 1" condim="3" friction="1 0.005 0.0001" density="1000" solmix="1" solimp="0.9 0.95 0.001" solref="0.02 1" margin="0" gap="0"/>
    </default>
    <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4"
        rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <!-- <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/> -->
        <texture type="2d" file="textures/woodenfloor.png" name="texplane"/>
        <material name="MatPlane" reflectance="0" shininess="0" specular="0" texrepeat="10 10" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>

    <asset>
    <mesh name="pyramid" file="stls/pyramid_centered.stl" scale="0.03 0.03 0.03" />
    </asset>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3"
            specular=".1 .1 .1" castshadow="true"/>
        <camera name="fixed" pos="0 0 6.1" quat="1 0 0 0" fovy="45"/>
        <camera name="external_camera_1" pos="0 -5.5 5.5" quat="0.9396 0.342 0 0" fovy="45"/>

        <geom name="floor" type="plane" material="MatPlane" pos=" 0  0  0" quat="1 0 0 0" size="{size} {size} 0.1" contype="1" conaffinity="1"/>

        <body name="agent" pos="0 0 0.11">
        <geom type="sphere" size="0.1" rgba="1 1 1 1" contype="1" conaffinity="1"/>
        <joint name="agent_slide_x" type="slide" axis="1 0 0" range="-3.0 3.0" damping="50"/>
        <joint name="agent_slide_y" type="slide" axis="0 1 0" range="-3.0 3.0" damping="50"/>
        </body>


        {cube_bodies}

        {cube_light_bodies}

        {cylinder_bodies}

        {pyramid_bodies}

        <body name="target" pos="{target_loc} {target_loc} 0.001">
            <geom type="cylinder" size="0.1 0.001" rgba="1 0 0 1" density="0.00001" contype="4" conaffinity="4"/>
            <site name="target" pos="0 0 0" size="0.002 0.002 0.002" rgba="1 0 0 1" type="sphere"></site>
        </body>

        <body name="walls" pos="0 0 0">
            <geom type="box" pos="{size} 0 0.1" size="0.1 {size_w_offset} 0.1" rgba="0.5 0.5 0.5 1" density="10000000" contype="1" conaffinity="3"/>
            <geom type="box" pos="-{size} 0 0.1" size="0.1 {size_w_offset} 0.1" rgba="0.5 0.5 0.5 1" density="10000000" contype="1" conaffinity="3"/>
            <geom type="box" pos="0 {size} 0.1" size="{size_w_offset} 0.1 0.1" rgba="0.5 0.5 0.5 1" density="10000000" contype="1" conaffinity="3"/>
            <geom type="box" pos="0 -{size} 0.1" size="{size_w_offset} 0.1 0.1" rgba="0.5 0.5 0.5 1" density="10000000" contype="1" conaffinity="3"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="agent_slide_x" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
        <motor joint="agent_slide_y" ctrlrange="-1 1" ctrllimited="true" gear="100"/>
    </actuator>
</mujoco>

        """


def generate_xml(num_cube=2, num_cube_light=1, num_cylinder=1, num_pyramid=1, playground_size=2.0, offset=0.1):

    assert playground_size >= 1.0, "Playground size can't be smaller than 1!"

    cube_base = """<body name="cube{id}" pos="0.3 0 0.1">
      <geom type="box" mass="2.0" size="0.1 0.1 0.1" rgba="0 1 0 1" contype="3" conaffinity="3"/>
        <!--joint name="cube{id}:joint" type="free" limited="false"/-->
      <joint name="cube{id}:slide_x" type="slide" pos="0 0 0" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="cube{id}:slide_y" type="slide" pos="0 0 0" axis="0 1 0" limited="false" damping="0.5"/>
      <joint name="cube{id}:hinge_z" type="hinge" pos="0 0 0" axis="0 0 1" limited="false" damping="0.5" stiffness="0" armature="0"/>
      <site name="cube{id}" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
    </body>"""

    cube_light_base = """<body name="cube_light{id}" pos="0.2 -0.5 0.1">
      <geom type="box" mass="0.8" size="0.1 0.1 0.1" rgba="0.8 0.2 0.8 1" contype="3" conaffinity="3"/>
      <!--joint name="cube_light{id}:joint" type="free" limited="false"/-->
      <joint name="cube_light{id}:slide_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="cube_light{id}:slide_y" type="slide" axis="0 1 0" limited="false" damping="0.5"/>
      <joint name="cube_light{id}:hinge_z" type="hinge" pos="0 0 0" axis="0 0 1" limited="false" damping="0.5" stiffness="0" armature="0"/>
    </body>"""

    cylinder_base = """<body name="cylinder{id}" pos="0.0 -1.0 0.06">
      <geom type="cylinder" mass="1.0" size="0.125 0.06" rgba="0.0 0.7 1.0 1" contype="3" conaffinity="3"/>
      <!--joint name="cylinder{id}:joint" type="free" limited="false"/-->
      <joint name="cylinder{id}:slide_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="cylinder{id}:slide_y" type="slide" axis="0 1 0" limited="false" damping="0.5"/>
      <joint name="cylinder{id}:hinge_z" type="hinge" pos="0 0 0" axis="0 0 1" limited="false" damping="0.5" stiffness="0" armature="0"/>
    </body>"""

    pyramid_base = """<body name="pyramid{id}" pos="=1.7 -1.7 0.3">
      <geom type="mesh" mesh="pyramid" mass="5.0" rgba="0.5 0.6 0.7 1" contype="3" conaffinity="3"/>
      <joint name="pyramid{id}:slide_x" type="slide" axis="1 0 0" limited="false" damping="100" />
      <joint name="pyramid{id}:slide_y" type="slide" axis="0 1 0" limited="false" damping="100" />
      <joint name="pyramid{id}:hinge_z" type="hinge" pos="0 0 0" axis="0 0 1" limited="false" damping="10" armature="0" />
      <site name="pyramid{id}" pos="0 0 0" size="0.015 0.015 0.015" rgba="1 0 0 0" type="sphere"></site>
    </body>"""

    # num_blocks = num_cube + num_cube_light + num_cylinder + num_pyramid
    cube = [cube_base.format(**dict(id=i)) for i in range(num_cube)]
    cube_light = [cube_light_base.format(**dict(id=i)) for i in range(num_cube_light)]
    cylinder = [cylinder_base.format(**dict(id=i)) for i in range(num_cylinder)]
    pyramid = [pyramid_base.format(**dict(id=i)) for i in range(num_pyramid)]

    return base.format(
        **dict(
            size=playground_size,
            size_w_offset=playground_size + offset,
            cube_bodies="\n".join(cube),
            cube_light_bodies="\n".join(cube_light),
            cylinder_bodies="\n".join(cylinder),
            pyramid_bodies="\n".join(pyramid),
            target_loc=playground_size - 0.3,
        )
    )


if __name__ == "__main__":
    xml_output = generate_xml()
    print(xml_output)
