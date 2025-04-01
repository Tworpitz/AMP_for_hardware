# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

MOTION_FILES = glob.glob("datasets/humanoid_mocap_motions/*")


class XYuanAMPCfg(HumanoidCfg):

    class env(HumanoidCfg.env):
        num_envs = 3000
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 75
        num_privileged_obs = 75
        reference_state_initialization = True
        reference_state_initialization_prob = 0.75
        amp_motion_files = MOTION_FILES

    class init_state(HumanoidCfg.init_state):
        pos = [0.0, 0.0, 1.03]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "yaw_joint": 0.0,
            # "roll_joint": 0,
            "arm_l1_joint": 0,
            "arm_l2_joint": 0,
            "arm_l3_joint": 0,
            "arm_l4_joint": 0,
            # "arm_l5_joint": 0,
            # "arm_l6_joint": 0,
            # "arm_l7_joint": 0,
            "arm_r1_joint": 0,
            "arm_r2_joint": 0,
            "arm_r3_joint": 0,
            "arm_r4_joint": 0,
            # "arm_r5_joint": 0,
            # "arm_r6_joint": 0,
            # "arm_r7_joint": 0,
            "leg_l1_joint": 0,
            "leg_l2_joint": 0,
            "leg_l3_joint": 0.2,
            "leg_l4_joint": -0.4,
            "leg_l5_joint": 0.2,
            "leg_l6_joint": 0,
            "leg_r1_joint": 0,
            "leg_r2_joint": 0,
            "leg_r3_joint": 0.2,
            "leg_r4_joint": -0.4,
            "leg_r5_joint": 0.2,
            "leg_r6_joint": 0
        }

    class control(HumanoidCfg.control):
        # PD Drive parameters:
        control_type = "P"
        # stiffness = {"joint":0}
        stiffness = {
            "yaw_joint": 100,
            # "roll_joint": 100,
            "arm_l1_joint": 100,
            "arm_l2_joint": 100,
            "arm_l3_joint": 100,
            "arm_l4_joint": 100,
            # "arm_l5_joint": 10,
            # "arm_l6_joint": 1,
            # "arm_l7_joint": 0.1,
            "arm_r1_joint": 100,
            "arm_r2_joint": 100,
            "arm_r3_joint": 100,
            "arm_r4_joint": 100,
            # "arm_r5_joint": 10,
            # "arm_r6_joint": 1,
            # "arm_r7_joint": 0.1,
            "leg_l1_joint": 400,
            "leg_l2_joint": 100,
            "leg_l3_joint": 100,
            "leg_l4_joint": 400,
            "leg_l5_joint": 100,
            "leg_l6_joint": 100,
            "leg_r1_joint": 400,
            "leg_r2_joint": 100,
            "leg_r3_joint": 100,
            "leg_r4_joint": 400,
            "leg_r5_joint": 100,
            "leg_r6_joint": 100
            }  # [N*m/rad]
        
        # damping = {"joint":0}
        # damping =  {
        #     "yaw_joint": 0.0,
        #     "roll_joint": 0,
        #     "arm_l1_joint": 0,
        #     "arm_l2_joint": 0,
        #     "arm_l3_joint": 0,
        #     "arm_l4_joint": 0,
        #     "arm_l5_joint": 0,
        #     "arm_l6_joint": 0,
        #     "arm_l7_joint": 0,
        #     "arm_r1_joint": 0,
        #     "arm_r2_joint": 0,
        #     "arm_r3_joint": 0,
        #     "arm_r4_joint": 0,
        #     "arm_r5_joint": 0,
        #     "arm_r6_joint": 0,
        #     "arm_r7_joint": 0,
        #     "leg_l1_joint": 0,
        #     "leg_l2_joint": 0,
        #     "leg_l3_joint": 0,
        #     "leg_l4_joint": 0,
        #     "leg_l5_joint": 0,
        #     "leg_l6_joint": 0,
        #     "leg_r1_joint": 0,
        #     "leg_r2_joint": 0,
        #     "leg_r3_joint": 0,
        #     "leg_r4_joint": 0,
        #     "leg_r5_joint": 0,
        #     "leg_r6_joint": 0
        #     }   # [N*m*s/rad]
        damping =  {
            "yaw_joint": 15.0,
            # "roll_joint": 10,
            "arm_l1_joint": 10,
            "arm_l2_joint": 10,
            "arm_l3_joint": 10,
            "arm_l4_joint": 1,
            # "arm_l5_joint": 0,
            # "arm_l6_joint": 0,
            # "arm_l7_joint": 0,
            "arm_r1_joint": 10,
            "arm_r2_joint": 10,
            "arm_r3_joint": 10,
            "arm_r4_joint": 1,
            # "arm_r5_joint": 0,
            # "arm_r6_joint": 0,
            # "arm_r7_joint": 0,
            "leg_l1_joint": 10,
            "leg_l2_joint": 4,
            "leg_l3_joint": 2,
            "leg_l4_joint": 4,
            "leg_l5_joint": 1,
            "leg_l6_joint": 1,
            "leg_r1_joint": 10,
            "leg_r2_joint": 4,
            "leg_r3_joint": 2,
            "leg_r4_joint": 4,
            "leg_r5_joint": 1,
            "leg_r6_joint": 1
            }   # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(HumanoidCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(HumanoidCfg.asset):
        # file = (
        #     "{LEGGED_GYM_ROOT_DIR}/resources/robots/xyuan/urdf/xyuan_description(with collision).urdf"
        # )
        file = (
            # "{LEGGED_GYM_ROOT_DIR}/resources/robots/xyuan/urdf/xyuan_description(with collision, 4-dof arm).urdf"
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/xyuan/urdf/xyuan_description(with collision, 4-dof arm, 1-dof waist).urdf"
        )
        foot_name = "foot"
        penalize_contacts_on = []  # 在包含这些字段的连杆上惩罚碰撞
        fix_base_link = False # fix the base of the robot
        terminate_after_contacts_on = ['leg_l1_link', 'leg_l2_link', 'leg_l3_link', #'leg_l4_link', #'leg_l5_link',
                                       'leg_r1_link', 'leg_r2_link', 'leg_r3_link', #'leg_r4_link', #'leg_r5_link', 
                                       'yaw_link', 'roll_link', 
                                       'arm_l1_link', 'arm_l2_link', 'arm_l3_link', 'arm_l4_link', 'arm_l5_link', 'arm_l6_link', 'arm_l7_link', 
                                       'arm_r1_link', 'arm_r2_link', 'arm_r3_link', 'arm_r4_link', 'arm_r5_link', 'arm_r6_link', 'arm_r7_link']
          # 在包含这些字段的连杆上终止
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        collapse_fixed_joints = True  # remove fixed joints from the URDF

    class domain_rand:
        randomize_friction = False
        friction_range = [0.25, 1.75]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards(HumanoidCfg.rewards):
        soft_dof_pos_limit = 0.9
        # base_height_target = 0.9
        only_positive_rewards = False
        tracking_sigma = 0.1

        class scales(HumanoidCfg.rewards.scales):
            termination = 100.0
            tracking_lin_vel = 50.0
            tracking_ang_vel = 20.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 5.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 1.5]  # min max [m/s]
            # lin_vel_x = [0.3, 1.0]  # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]


class XYuanAMPCfgPPO(HumanoidCfgPPO):
    runner_class_name = "XyuanAMPOnPolicyRunner"

    class algorithm(HumanoidCfgPPO.algorithm):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        # learning_rate = 1e-3
        disc_coef = 5.0

    class runner(HumanoidCfgPPO.runner):
        run_name = ""
        experiment_name = "xyuan_amp_example"
        algorithm_class_name = "AMPPPO"
        policy_class_name = "ActorCritic"
        max_iterations = 500000  # number of policy updates

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        
        disc_grad_penalty = 5.0

        min_normalized_std = [0.02] * 21
